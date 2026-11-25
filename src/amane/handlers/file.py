import errno
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from ..config import HotSettings, WatermarkConfig
from ..db.models import Library, MediaFile, Metadata
from ..db.repo_types import MediaFileUpdates
from ..enums import ActorGender, DownloadableResource, LinkMode
from ..library import MEDIA_EXTENSIONS, LibraryFileKind, LibraryScan, ancestor_dirs, prune_empty_dirs
from ..library.rules import TRASH_DIRNAME, is_in_trash
from ..media import ResourceStore, allows_poster_crop, apply_cover_watermarks_from_info, crop_poster
from ..media import write_nfo as write_nfo_file
from ..media.resource_store import internal_url_hash
from ..net.http import WebClient
from ..observability.models import OrganizeConflict, OrganizeConflictReason
from ..organize import (
    EXTRAFANART_NAME_TEMPLATE_DEFAULT,
    MoveMode,
    PlaceOutcome,
    ResolvedPaths,
    SubtitleConflict,
    TargetState,
    discover_subtitles,
    execute_organize,
    place_subtitles,
    render_extrafanart_names,
    render_strm_content,
    resolve_paths,
    target_state,
    video_dest,
)
from ..organize.link import create_video_link
from ..parsing import ContentType, FileInfo, parse_file_info
from ..utils.path import existing_disk_path as existing_disk_path_sync
from ..utils.path import is_descendant, nfc_path, path_is_under
from ..utils.threads import existing_disk_path, in_thread, path_is_dir
from ._common import LibraryTaskLocks, refresh_external_subtitle
from .models import (
    ORGANIZE_CONFLICT_LIMIT,
    CleanupPayload,
    CleanupResult,
    OrganizePayload,
    OrganizeResult,
)
from .protocol import TaskHandler, TaskResult

if TYPE_CHECKING:
    from ..config import HotSettings
    from ..db.models import Metadata
    from ..db.repository import Repository
    from ..media import ResourceStore
    from ..net.http import WebClient

logger = structlog.get_logger()


@dataclass
class FileOperationsResult:
    outcome: PlaceOutcome
    dest: Path | None = None
    """PLACED 是落盘路径; FAILED 时可能仍带路径, 以便回写 MediaFile.path 补链接."""
    conflict_target: Path | None = None
    """CONFLICT 时被占用的目标路径."""
    subtitle_conflicts: list[SubtitleConflict] = field(default_factory=list)
    error: str | None = None


async def execute_file_operations(
    media_file: MediaFile,
    metadata: Metadata,
    paths: ResolvedPaths,
    move_mode: MoveMode,
    resource_store: ResourceStore,
    file_info: FileInfo,
    download_images: bool = True,
    write_nfo: bool = True,
    copy_resources: Sequence[DownloadableResource] | None = None,
    web_client: WebClient | None = None,
    config: HotSettings | None = None,
    library: Library | None = None,
    safe_dirs: Sequence[Path] | None = (),
    watermark_dir: Path | None = None,
    actor_genders: dict[str, ActorGender] | None = None,
) -> FileOperationsResult:
    """``file_info`` 由渲染路径的那一次 ``parse_file_info`` 产出, 必须带上 ``has_external_subtitle``.

    未带库列时函数不报错, 直接少打中字角标 — 调用方负责构造.
    """
    source_path = await existing_disk_path(Path(media_file.path))
    if source_path is None:
        logger.warning("source file missing", path=media_file.path)
        return FileOperationsResult(outcome=PlaceOutcome.FAILED, error=f"源文件不存在: {media_file.path}")

    # 目标被别的文件占用时立即返回: 不下载图片, 不写附属文件, 不改库内任何路径.
    # 读不出目标状态时同样立即返回, 但记为失败而不是冲突.
    # 目标路径必须与 execute_organize 同一算法: 模板写死扩展名时, 模板渲染结果与真实目标不同.
    dest = video_dest(paths.video.parent, paths.video.stem, source_path)
    check = await target_state(source_path, dest)
    if check.state is TargetState.OCCUPIED:
        logger.warning("organize target occupied", source=str(source_path), dest=str(dest))
        return FileOperationsResult(outcome=PlaceOutcome.CONFLICT, conflict_target=dest)
    if check.state is TargetState.UNKNOWN:
        logger.warning("organize target state unknown", source=str(source_path), dest=str(dest), error=check.error)
        return FileOperationsResult(outcome=PlaceOutcome.FAILED, error=check.error)

    # 下载图片; 水印打在库路径副本上, 不能修改 Resource 原图.
    if web_client and download_images:
        logger.debug("downloading images via store", number=metadata.number)
        kinds = set(copy_resources) if copy_resources is not None else set(DownloadableResource)
        await _download_images_via_store(
            metadata,
            resource_store,
            web_client,
            paths,
            config,
            kinds,
            file_info=file_info,
            content_type=metadata.content_type,
            watermark_dir=watermark_dir,
            library=library,
        )

    # 发现字幕: 必须在视频移动前检查同目录.
    subtitles: list[Path] = []
    if library is not None:
        subtitles = await discover_subtitles(source_path, library.subtitle_extensions, file_info)

    org_result = await execute_organize(
        source=source_path,
        target_dir=paths.video.parent,
        target_stem=paths.video.stem,
        mode=move_mode,
    )

    if org_result.outcome is PlaceOutcome.CONFLICT:
        return FileOperationsResult(outcome=PlaceOutcome.CONFLICT, conflict_target=org_result.dest)

    # 写链接; 失败仍带 dest, 以便回写 MediaFile.path.
    if org_result.outcome is PlaceOutcome.PLACED and org_result.dest and paths.link is not None:
        mode = LinkMode(library.link_mode) if library is not None else LinkMode.STRM
        strm_content: str | None = None
        if mode == LinkMode.STRM and library is not None:
            try:
                strm_content = render_strm_content(
                    library.strm_content_template,
                    org_result.dest,
                    Path(library.path),
                    metadata,
                    source_path=source_path,
                    file_info=file_info,
                    link=paths.link,
                    actor_genders=actor_genders,
                )
            except ValueError as e:
                return FileOperationsResult(outcome=PlaceOutcome.FAILED, dest=org_result.dest, error=str(e))
        link_result = await create_video_link(org_result.dest, paths.link, mode, content=strm_content)
        if link_result.outcome is not PlaceOutcome.PLACED:
            return FileOperationsResult(outcome=PlaceOutcome.FAILED, dest=org_result.dest, error=link_result.error)

    # 写入 NFO.
    if org_result.outcome is PlaceOutcome.PLACED and org_result.dest and write_nfo:
        await write_nfo_file(metadata, paths.nfo, mosaic=file_info.mosaic)

    # 字幕按模板落到 video_dest 侧.
    subtitle_conflicts: list[SubtitleConflict] = []
    if org_result.outcome is PlaceOutcome.PLACED and org_result.dest and library is not None and subtitles:
        subtitle_conflicts = await place_subtitles(
            subtitles,
            video_source=source_path,
            video_dest=org_result.dest,
            library=library,
            metadata=metadata,
            file_info=file_info,
            mode=move_mode,
            safe_dirs=safe_dirs,
            link_dir=paths.link.parent if paths.link is not None else None,
            link_name=paths.link.stem if paths.link is not None else None,
            actor_genders=actor_genders,
        )

    if org_result.outcome is PlaceOutcome.PLACED:
        logger.debug("file operations done", number=metadata.number, dest=str(org_result.dest), mode=str(move_mode))
        return FileOperationsResult(
            outcome=PlaceOutcome.PLACED, dest=org_result.dest, subtitle_conflicts=subtitle_conflicts
        )
    logger.debug("file operations failed", number=metadata.number, error=org_result.error)
    return FileOperationsResult(outcome=PlaceOutcome.FAILED, error=org_result.error)


async def apply_file_operations(
    repo: Repository,
    media_file_id: int | None,
    metadata: Metadata,
    config: HotSettings,
    resource_store: ResourceStore,
    *,
    write_nfo: bool = True,
    copy_resources: Sequence[DownloadableResource] | None = None,
    web_client: WebClient | None = None,
    safe_dirs: Sequence[Path] | None = (),
    watermark_dir: Path | None = None,
) -> FileOperationsResult | None:
    """缺少 media_file_id 或对应记录时返回 None (跳过, 不视为失败).
    Library 由 MediaFile.library_id 派生.
    """
    if media_file_id is None:
        return None
    media_file = await repo.get_media_file(media_file_id)
    if media_file is None:
        return None
    library = await repo.get_library(media_file.library_id)
    if library is None:
        return None

    # 渲染路径后执行落盘.
    ext = Path(media_file.path).suffix.lstrip(".")
    file_info = parse_file_info(media_file.path, has_external_subtitle=media_file.has_external_subtitle)
    actor_genders = {a.name: a.gender for a in await repo.get_actors_by_names(metadata.actors)}
    paths = resolve_paths(
        library,
        metadata,
        ext=ext,
        file_info=file_info,
        source_path=Path(media_file.path),
        safe_dirs=safe_dirs,
        actor_genders=actor_genders,
    )
    return await execute_file_operations(
        media_file,
        metadata,
        paths,
        library.move_mode,
        resource_store,
        file_info,
        download_images=True,
        write_nfo=write_nfo,
        copy_resources=copy_resources,
        web_client=web_client,
        config=config,
        library=library,
        safe_dirs=safe_dirs,
        watermark_dir=watermark_dir,
        actor_genders=actor_genders,
    )


async def commit_organized_media_file(
    repo: Repository,
    media: MediaFile,
    placed: Path,
    library_root: Path,
) -> None:
    """整理后的路径仍在本库内才改 path; 已离开本库且源路径不在磁盘上则删行.

    目标路径已被另一行占用时删本行; 占用行缺少刮削字段则从本行补上.
    """
    if media.id is None:
        return
    library = await repo.get_library(media.library_id)
    if not is_descendant(placed, library_root):
        if await existing_disk_path(Path(media.path), follow_symlinks=False) is None:
            await repo.delete_media_file(media.id)
        return

    occupant = await repo.get_media_file_by_path(str(placed))
    if occupant is None or occupant.id == media.id:
        updated = await repo.update_media_file(media.id, path=str(placed))
        if updated is not None and library is not None:
            await refresh_external_subtitle(repo, updated, library)
        return
    if occupant.id is None:
        return
    occupant_updates: MediaFileUpdates = {}
    if occupant.metadata_id is None and media.metadata_id is not None:
        occupant_updates["metadata_id"] = media.metadata_id
        occupant_updates["status"] = media.status
    if occupant.oshash is None and media.oshash is not None:
        occupant_updates["oshash"] = media.oshash
    if occupant.number is None and media.number is not None:
        occupant_updates["number"] = media.number
    if occupant_updates:
        await repo.update_media_file(occupant.id, **occupant_updates)
    await repo.delete_media_file(media.id)
    if library is not None:
        await refresh_external_subtitle(repo, occupant, library)


async def _resolve_local(url: str, store: ResourceStore, client: WebClient) -> Path | None:
    """内部 `/api/resources/{hash}` 与外部 URL 均由 store 解析 (命中缓存直出)."""
    return await store.acquire(url, client)


async def _local_source(url: str, store: ResourceStore) -> Path | None:
    """只解析本地文件, 不下载也不重新派生: 判定阶段用它对齐落盘阶段的解析口径.

    手动裁切把 ``poster_urls`` 换成内部 URL, 落盘走 ``acquire`` 能命中那张裁切图;
    判定也必须按 hash 查到同一个文件, 否则会以为产物缺失并反复重建.
    """
    token = internal_url_hash(url)
    if token is not None:
        found = await store.get_by_url_hash(token)
        return found[1] if found is not None else None
    return await store.resolve(url)



async def _acquire_first_local(urls: list[str], store: ResourceStore, client: WebClient) -> Path | None:
    for url in urls:
        path = await _resolve_local(url, store, client)
        if path:
            return path
    return None


# 部分 NAS 的存储层 (网络重挂载、回环文件系统) 会在删除风暴后让 rename 短暂失败,
# 退避重试通常就能过去; 仍失败则退化为直接覆盖写, 不再依赖 rename.
_REPLACE_RETRY_ERRNOS = frozenset({errno.EBADMSG, errno.EBUSY, errno.EAGAIN, errno.EINTR, errno.ETXTBSY})
_REPLACE_ATTEMPTS = 3
_REPLACE_BACKOFF_SECONDS = 0.4


def _replace_with_retry(tmp: Path, target: Path) -> None:
    """把临时文件改名到目标; 存储层抖动时退避重试, 最终退化为直接覆盖写."""
    delay = _REPLACE_BACKOFF_SECONDS
    last: OSError | None = None
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            tmp.replace(target)
            return
        except OSError as exc:
            if exc.errno not in _REPLACE_RETRY_ERRNOS:
                raise
            last = exc
            logger.warning("library rename retried", target=str(target), attempt=attempt, error=str(exc))
            time.sleep(delay)
            delay *= 2
    try:
        shutil.copyfile(tmp, target)
    except OSError as exc:
        logger.error("library write failed", target=str(target), error=str(exc))
        raise
    logger.warning("library write fell back to overwrite", target=str(target), error=str(last))


def _write_into_library(source: Path, target: Path) -> None:
    """按新文件落盘库内产物: 模式位与 ACL 由目标目录决定, 不继承源文件.

    源在 data 目录, 那里通常只授予容器用户访问; 带上源的元数据会把这张 ACL 搬进媒体库,
    媒体服务器因此读不到产物. 临时文件加改名也让目标已存在时同样按新文件重写, 重跑整理
    即可修好旧产物.
    """
    tmp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, tmp)
    except OSError as exc:
        logger.error("library write failed", source=str(source), target=str(target), error=str(exc))
        raise
    try:
        _replace_with_retry(tmp, target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("temp file cleanup skipped", path=str(tmp), error=str(exc))


_FIXED_PRODUCT_NAMES = frozenset({"poster.jpg", "thumb.jpg", "fanart.jpg", "trailer.mp4", "movie.nfo", "tvshow.nfo"})
# 判定「条目有没有变化」时只把这几类后缀算作本工具的产物; 字幕等由整理另写, 不参与判定.
_MANAGED_SUFFIXES = MEDIA_EXTENSIONS | {".nfo"}


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _starts_with_number(name: str, number: str) -> bool:
    """名字是否以该番号开头且后一位不是数字; 番号是 LTX-123 这类前缀时不会误命中间缀."""
    return re.match(rf"{re.escape(number)}(?![0-9])", name, re.IGNORECASE) is not None


def _is_dedicated_dir(directory: Path, number: str) -> bool:
    """目录名是否就是本条目: 只有默认模板 ``{studio}/{number}/`` 这种按番号命名的目录才算.

    摊平到工作室目录时目录里放着别的番号, 一律只删番号前缀文件; 目录里还有子目录时,
    那些子目录可能是别的条目, 既不递归也不整目录清空.
    """
    folded = number.casefold()
    name = directory.name.casefold()
    return name == folded or name.startswith(f"{folded} ")


def _same_path(left: Path, right: Path) -> bool:
    return path_is_under(left, right) and path_is_under(right, left)


def _affects_rebuild(path: Path, extrafanart_dirs: Sequence[Path]) -> bool:
    """这个将被清掉的路径是否说明条目变了; 别的软件写的附属文件 (mediainfo 等) 不算."""
    if any(path_is_under(path, directory) for directory in extrafanart_dirs):
        return True
    return path.suffix.casefold() in _MANAGED_SUFFIXES or path.name.casefold() in _FIXED_PRODUCT_NAMES


def _epoch(moment: datetime) -> float:
    """datetime 归一成 epoch 秒; ``Metadata.updated_at`` 可能带时区, 比较只用数字就不必猜口径."""
    return moment.replace(tzinfo=UTC).timestamp() if moment.tzinfo is None else moment.timestamp()


def _image_products(
    paths: ResolvedPaths,
    sources: dict[DownloadableResource, Path],
    *,
    content_type: ContentType,
    config: HotSettings,
) -> set[Path]:
    """本轮会写出的封面 / 海报 / 剧照 / 预告片路径; 没有源的一律不计, 免得永远判成「缺文件」."""
    products: set[Path] = set()
    if DownloadableResource.thumb in sources:
        products.add(_resolved(paths.thumb))
        products.add(_resolved(paths.fanart))
    poster = DownloadableResource.poster in sources
    if not poster and DownloadableResource.thumb in sources:
        poster = config.scraping.crop_poster and allows_poster_crop(content_type)
    if poster:
        products.add(_resolved(paths.poster))
    if DownloadableResource.trailer in sources:
        products.add(_resolved(paths.trailer))
    return products


@dataclass(frozen=True, slots=True)
class _RebuildPlan:
    """一个条目的清理计划: 留到该条目真正落盘前才执行, 避免整库范围先删后写."""

    number: str
    directories: tuple[Path, ...]
    extrafanart_dirs: tuple[Path, ...]
    keep: frozenset[Path]
    expected: frozenset[Path]


async def _item_needs_rebuild(
    expected: set[Path],
    products: set[Path],
    keep: set[Path],
    directories: Sequence[Path],
    metadata: Metadata,
    library: Library,
    *,
    sources: dict[DownloadableResource, Path],
    stills: Sequence[Path],
    extrafanart_dir: Path | None,
    extrafanart_dirs: Sequence[Path] = (),
) -> bool:
    """条目是否需要重建: 有残留、缺产物、或产物比 metadata / 源资源旧.

    残留判定直接复用 dry-run 的清理候选; 因此「目录里多了上一轮的文件」一定算变化, 而 mediainfo
    这类外部工具写的附属文件不参与判定 — 它们只在该条目因别的原因重建时被顺带删除.
    ``expected`` 含视频落点, 只用于判「缺文件」; 新旧比较只看 ``products`` (NFO / 链接 /
    图片 / 剧照) — 视频不会因整理被重写, 其 mtime 是下载时间, 混进来会让每个条目都判成过期.
    """
    if not expected:
        return True
    stale = await _purge_item_dirs(
        directories,
        metadata.number,
        keep | expected,
        extrafanart_dirs=extrafanart_dirs,
        library_root=Path(library.path),
        preserve=expected,
        dry_run=True,
    )
    if any(_affects_rebuild(path, extrafanart_dirs) for path in stale):
        return True
    for path in expected:
        try:
            path.stat()
        except OSError:
            return True
    oldest: float | None = None
    for path in products:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return True
        oldest = mtime if oldest is None else min(oldest, mtime)
    stamp = _epoch(metadata.updated_at)
    for source in sources.values():
        try:
            stamp = max(stamp, source.stat().st_mtime)
        except OSError:
            return True
    if stills:
        if extrafanart_dir is None:
            return True
        try:
            files = [child for child in extrafanart_dir.iterdir() if child.is_file()]
        except OSError:
            return True
        if len(files) != len(stills):
            return True
        for child in files:
            oldest = child.stat().st_mtime if oldest is None else min(oldest, child.stat().st_mtime)
    return oldest is not None and stamp > oldest


@in_thread
def _purge_item_dirs(
    directories: Sequence[Path],
    number: str,
    keep: set[Path],
    *,
    extrafanart_dirs: Sequence[Path] = (),
    library_root: Path | None = None,
    preserve: set[Path] | frozenset[Path] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """清空这些目录里属于本条目的东西; 返回被删 (或将被删) 的路径.

    条目独占目录 (目录名就是番号) 整目录清空: 旧 CD / 集数文件、剧照、NFO、链接, 连同别的工具
    写进来的 mediainfo 等附属文件一并删除, 由重建后的写入补齐. 摊平到共享目录时只删名字以番号
    开头的文件, 别的条目一个字节都不碰. ``keep`` (仍在盘上的成员源文件) 始终保留,
    库根与回收目录一律跳过. ``dry_run`` 只列候选, 不落删, 供「条目有没有变化」的判定复用.
    """
    removed: list[Path] = []
    seen: set[Path] = set()
    dropped: set[Path] = set()
    preserved = preserve or frozenset()
    trash_root = library_root / TRASH_DIRNAME if library_root is not None else None

    def protected(path: Path) -> bool:
        if _resolved(path) in keep:
            return True
        if library_root is not None:
            if _same_path(path, library_root):
                return True
            if trash_root is not None and path_is_under(path, trash_root):
                return True
        return False

    def drop(path: Path) -> None:
        key = _resolved(path)
        if key in dropped:
            return
        dropped.add(key)
        if path.is_dir() and not path.is_symlink():
            try:
                children = list(path.iterdir())
            except OSError as exc:
                logger.warning("rebuild cleanup skipped", path=str(path), error=str(exc))
                return
            for child in children:
                if not protected(child):
                    drop(child)
            if not dry_run:
                try:
                    if not any(path.iterdir()):
                        path.rmdir()
                except OSError as exc:
                    logger.debug("rebuild cleanup kept directory", path=str(path), error=str(exc))
            return
        removed.append(path)
        if dry_run:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("rebuild cleanup skipped", path=str(path), error=str(exc))

    def purge(directory: Path, *, whole_dir: bool) -> None:
        key = _resolved(directory)
        if key in seen or not directory.is_dir() or protected(directory):
            return
        seen.add(key)
        try:
            children = list(directory.iterdir())
        except OSError as exc:
            logger.warning("rebuild cleanup skipped", path=str(directory), error=str(exc))
            return
        for child in children:
            if protected(child):
                continue
            if (
                whole_dir
                and child.is_file()
                and child.name.casefold() in _FIXED_PRODUCT_NAMES
                and _resolved(child) in preserved
            ):
                # 本轮确实要重写的固定名产物不预删 (写入时直接覆盖), 免得删了却没写回来.
                continue
            if not whole_dir and (child.is_dir() or not _starts_with_number(child.name, number)):
                continue
            drop(child)

    for directory in directories:
        purge(directory, whole_dir=_is_dedicated_dir(directory, number))
    for directory in extrafanart_dirs:
        purge(directory, whole_dir=True)
    return removed


@in_thread
def _place_library_images(
    paths: ResolvedPaths,
    *,
    thumb_local: Path | None,
    poster_local: Path | None,
    trailer_local: Path | None,
    extrafanart_targets: Sequence[tuple[str, Path]],
    kinds: set[DownloadableResource],
    config: HotSettings | None,
    file_info: FileInfo | None,
    content_type: ContentType,
    watermark_dir: Path | None,
) -> None:
    # 复制封面 / 海报 / 预告片 / extrafanart 到库路径, 不判断目标上是否已有产物.
    # fanart 没有刮削来源, 内容取 thumb, 随 thumb 一同重写.
    if thumb_local is not None and DownloadableResource.thumb in kinds:
        paths.thumb.parent.mkdir(parents=True, exist_ok=True)
        _write_into_library(thumb_local, paths.thumb)
        paths.fanart.parent.mkdir(parents=True, exist_ok=True)
        _write_into_library(thumb_local, paths.fanart)

    if DownloadableResource.poster in kinds:
        if poster_local is not None:
            paths.poster.parent.mkdir(parents=True, exist_ok=True)
            _write_into_library(poster_local, paths.poster)
        elif (
            thumb_local is not None
            and config is not None
            and config.scraping.crop_poster
            and allows_poster_crop(content_type)
        ):
            paths.poster.parent.mkdir(parents=True, exist_ok=True)
            crop_poster(
                paths.thumb,
                paths.poster,
                poster_ratio=config.scraping.poster_ratio,
                jpeg_quality=config.scraping.jpeg_quality,
            )

    if trailer_local is not None and DownloadableResource.trailer in kinds:
        paths.trailer.parent.mkdir(parents=True, exist_ok=True)
        _write_into_library(trailer_local, paths.trailer)

    if extrafanart_targets and DownloadableResource.extrafanart in kinds:
        paths.extrafanart_dir.mkdir(parents=True, exist_ok=True)
        for name, p in extrafanart_targets:
            _write_into_library(p, paths.extrafanart_dir / name)

    # 叠加水印 (thumb / poster).
    wm = config.watermark if config is not None else WatermarkConfig()
    if wm.enabled and file_info is not None:
        jpeg_quality = config.scraping.jpeg_quality if config is not None else 95
        if paths.thumb.exists():
            apply_cover_watermarks_from_info(
                paths.thumb,
                file_info,
                content_type,
                jpeg_quality=jpeg_quality,
                watermark_dir=watermark_dir,
                scale=wm.scale,
                corners=wm.corners,
            )
        if paths.poster.exists():
            apply_cover_watermarks_from_info(
                paths.poster,
                file_info,
                content_type,
                jpeg_quality=jpeg_quality,
                watermark_dir=watermark_dir,
                scale=wm.scale,
                corners=wm.corners,
            )


async def _download_images_via_store(
    metadata: Metadata,
    store: ResourceStore,
    client: WebClient,
    paths: ResolvedPaths,
    config: HotSettings | None,
    kinds: set[DownloadableResource],
    file_info: FileInfo | None,
    content_type: ContentType,
    watermark_dir: Path | None = None,
    library: Library | None = None,
) -> None:
    """优先用 Resource 已有文件; 缺失则现场 acquire.

    剧照文件名在这里渲染: 补零宽度取决于本轮剧照数量, 只有拿到文件列表后才定得下来.
    """
    thumb_local = None
    if DownloadableResource.thumb in kinds:
        thumb_urls = metadata.thumb_urls
        thumb_local = await _acquire_first_local(thumb_urls, store, client) if thumb_urls else None

    poster_local = None
    if DownloadableResource.poster in kinds:
        poster_urls = metadata.poster_urls
        poster_local = await _acquire_first_local(poster_urls, store, client) if poster_urls else None

    trailer_local = None
    if DownloadableResource.trailer in kinds:
        trailer_urls = metadata.trailer_urls
        trailer_local = await _acquire_first_local(trailer_urls, store, client) if trailer_urls else None

    extrafanart: list[Path] = []
    if DownloadableResource.extrafanart in kinds:
        extrafanart_by_site = metadata.extrafanart_urls
        if extrafanart_by_site:
            priority = list(extrafanart_by_site.keys())
            extrafanart = await store.acquire_extrafanart(extrafanart_by_site, priority, client)

    # 名字与源文件成对落地: 文件名模板只能在这里渲染, 宽度取决于本轮张数.
    extrafanart_targets: list[tuple[str, Path]] = []
    if extrafanart:
        template = library.extrafanart_name_template if library is not None else None
        names = render_extrafanart_names(template or EXTRAFANART_NAME_TEMPLATE_DEFAULT, paths.context, len(extrafanart))
        extrafanart_targets = list(zip(names, extrafanart, strict=True))

    await _place_library_images(
        paths,
        thumb_local=thumb_local,
        poster_local=poster_local,
        trailer_local=trailer_local,
        extrafanart_targets=extrafanart_targets,
        kinds=kinds,
        config=config,
        file_info=file_info,
        content_type=content_type,
        watermark_dir=watermark_dir,
    )


@in_thread
def _classify_indexed(path: Path, scan: LibraryScan) -> LibraryFileKind | None:
    return scan.classify(path)


class OrganizeHandler(TaskHandler[OrganizePayload, OrganizeResult]):
    """依据已有 Metadata 整理范围内的 MediaFile; 不刮削, 不修改 Metadata, 不扫描磁盘.

    同库执行期与 DELETE 共用一把锁.
    """

    def __init__(
        self,
        repo: Repository,
        config: HotSettings,
        resource_store: ResourceStore,
        web_client: WebClient | None = None,
        safe_dirs: Sequence[Path] | None = (),
        watermark_dir: Path | None = None,
        *,
        library_locks: LibraryTaskLocks | None = None,
    ):
        super().__init__(payload_t=OrganizePayload, result_t=OrganizeResult)
        self._repo = repo
        self._config = config
        self._web_client = web_client
        self._resource_store = resource_store
        self._safe_dirs = safe_dirs
        self._watermark_dir = watermark_dir
        self._library_locks = library_locks if library_locks is not None else LibraryTaskLocks()

    async def _load_scope(self, payload: OrganizePayload, library: Library) -> list[MediaFile]:
        assert library.id is not None
        if payload.media_file_ids is not None:
            return await self._repo.list_media_files(library_id=library.id, ids=payload.media_file_ids, limit=None)
        indexed = await self._repo.list_media_files(library_id=library.id, limit=None)
        if nfc_path(payload.path) != nfc_path(library.path):
            return [mf for mf in indexed if path_is_under(mf.path, payload.path)]
        return indexed

    async def _product_sources(
        self,
        metadata: Metadata,
        kinds: set[DownloadableResource],
    ) -> tuple[dict[DownloadableResource, Path], list[Path]]:
        """取本轮会用来落盘的本地文件; 只解析已有资源, 不发起下载.

        判定「条目有没有变化」时, 资源缺失意味着产物本来就写不出来, 不该据此判成变化.
        """
        sources: dict[DownloadableResource, Path] = {}
        for kind, urls in (
            (DownloadableResource.thumb, metadata.thumb_urls),
            (DownloadableResource.poster, metadata.poster_urls),
            (DownloadableResource.trailer, metadata.trailer_urls),
        ):
            if kind not in kinds:
                continue
            for url in urls:
                path = await _local_source(url, self._resource_store)
                if path is not None:
                    sources[kind] = path
                    break
        stills: list[Path] = []
        if DownloadableResource.extrafanart in kinds:
            for urls in metadata.extrafanart_urls.values():
                resolved = [await _local_source(url, self._resource_store) for url in urls]
                if resolved and all(path is not None for path in resolved):
                    stills = [path for path in resolved if path is not None]
                    break
        return sources, stills

    async def _rebuild_items(
        self,
        live: list[MediaFile],
        library: Library,
        payload: OrganizePayload,
    ) -> tuple[list[MediaFile], dict[int, _RebuildPlan]]:
        """筛出需要重建的条目, 为每个条目准备清理计划, 并把条目全部分集纳入本次整理.

        整理是按番号执行的: 一个条目有 CD1 / CD2 两个源文件时, 任一变化都要把该条目的目录清空,
        再把两个分集一起补回去, 所以 ``rebuilt_members`` 收集的是条目下所有仍在盘上的成员.
        ``keep`` 只保留这些成员自己的源文件; ``expected`` 是本轮应该写出的全部文件,
        ``products`` 是其中会被重写的那部分. 清理计划留到该条目真正落盘前才执行.
        """
        metadata_ids = sorted({mf.metadata_id for mf in live if mf.metadata_id is not None})
        plans: dict[int, _RebuildPlan] = {}
        if not metadata_ids:
            return live, plans
        members_by_metadata: dict[int, list[MediaFile]] = {}
        for member in await self._repo.list_media_files(metadata_ids=metadata_ids, limit=None):
            if member.metadata_id is not None:
                members_by_metadata.setdefault(member.metadata_id, []).append(member)
        scoped = {mf.metadata_id for mf in live}
        write_nfo = library.write_nfo if payload.write_nfo is None else payload.write_nfo
        copy_resources = library.copy_resources if payload.copy_resources is None else payload.copy_resources
        kinds = set(copy_resources) if copy_resources is not None else set(DownloadableResource)
        unchanged: set[int] = set()
        rebuilt_members: dict[int, list[MediaFile]] = {}
        for metadata_id, members in members_by_metadata.items():
            if metadata_id not in scoped:
                continue
            metadata = await self._repo.get_metadata(metadata_id)
            if metadata is None:
                continue
            keep: set[Path] = set()
            expected: set[Path] = set()
            products: set[Path] = set()
            directories: list[Path] = []
            extrafanart_dirs: list[Path] = []
            live_paths: ResolvedPaths | None = None
            present: list[MediaFile] = []
            for member in members:
                source = await existing_disk_path(Path(member.path), follow_symlinks=False)
                if source is None:
                    continue
                present.append(member)
                keep.add(_resolved(source))
                ext = Path(member.path).suffix.lstrip(".")
                file_info = parse_file_info(member.path, has_external_subtitle=member.has_external_subtitle)
                paths = resolve_paths(
                    library,
                    metadata,
                    ext=ext,
                    file_info=file_info,
                    source_path=Path(member.path),
                    safe_dirs=self._safe_dirs,
                )
                directories.append(paths.video.parent)
                expected.add(
                    _resolved(video_dest(paths.video.parent, paths.video.stem, source, suffix=paths.video.suffix))
                )
                if paths.link is not None:
                    products.add(_resolved(paths.link))
                    directories.append(paths.link.parent)
                if write_nfo:
                    products.add(_resolved(paths.nfo))
                if DownloadableResource.extrafanart in kinds:
                    extrafanart_dirs.append(paths.extrafanart_dir)
                live_paths = live_paths or paths
            sources, stills = await self._product_sources(metadata, kinds)
            if live_paths is not None:
                products.update(
                    _image_products(
                        live_paths,
                        sources,
                        content_type=metadata.content_type,
                        config=self._config,
                    )
                )
            expected.update(products)
            if live_paths is not None and stills:
                extrafanart_vars = live_paths.context
                template = library.extrafanart_name_template or EXTRAFANART_NAME_TEMPLATE_DEFAULT
                for name in render_extrafanart_names(template, extrafanart_vars, len(stills)):
                    expected.add(_resolved(live_paths.extrafanart_dir / name))
            if not expected:
                # 库里没有可落盘的成员 (换版后旧行指向的文件已消失): 什么都不写, 更不能清目录.
                unchanged.add(metadata_id)
                continue
            if not await _item_needs_rebuild(
                expected,
                products,
                keep,
                directories,
                metadata,
                library,
                sources=sources,
                stills=stills,
                extrafanart_dir=live_paths.extrafanart_dir if live_paths is not None else None,
                extrafanart_dirs=tuple(extrafanart_dirs),
            ):
                unchanged.add(metadata_id)
                continue
            plans[metadata_id] = _RebuildPlan(
                number=metadata.number,
                directories=tuple(dict.fromkeys(_resolved(directory) for directory in directories)),
                extrafanart_dirs=tuple(dict.fromkeys(_resolved(directory) for directory in extrafanart_dirs)),
                keep=frozenset(keep),
                expected=frozenset(expected),
            )
            rebuilt_members[metadata_id] = present
        if payload.rebuild_limit and len(plans) > payload.rebuild_limit:
            order: list[int] = []
            for media_file in live:
                metadata_id = media_file.metadata_id
                if metadata_id is not None and metadata_id in plans and metadata_id not in order:
                    order.append(metadata_id)
            allowed = set(order[: payload.rebuild_limit])
            for metadata_id in [key for key in plans if key not in allowed]:
                plans.pop(metadata_id)
                unchanged.add(metadata_id)
            logger.info("organize rebuild limited", limit=payload.rebuild_limit, deferred=len(order) - len(allowed))
        if unchanged:
            logger.info("organize rebuild skipped unchanged items", count=len(unchanged))
        ordered = [mf for mf in live if mf.metadata_id not in unchanged]
        seen = {mf.id for mf in ordered}
        for members in rebuilt_members.values():
            for member in members:
                if member.id is not None and member.id not in seen:
                    seen.add(member.id)
                    ordered.append(member)
        return ordered, plans

    async def handle(self, payload: OrganizePayload) -> TaskResult[OrganizeResult]:
        library = await self._repo.get_library(payload.library_id)
        if library is None:
            return TaskResult(success=False, error=f"媒体库 {payload.library_id} 不存在")
        assert library.id is not None
        library_root = Path(library.path)
        if not await path_is_dir(library_root):
            return TaskResult(success=False, error=f"不是目录: {library.path}")
        if payload.media_file_ids is None:
            scope = Path(payload.path) if payload.path else library_root
            if nfc_path(str(scope)) != nfc_path(library.path) and not await path_is_dir(scope):
                return TaskResult(success=False, error=f"不是目录: {scope}")

        lock = await self._library_locks.get(library.id)
        async with lock:
            return await self._handle_unlocked(payload, library)

    async def _handle_unlocked(self, payload: OrganizePayload, library: Library) -> TaskResult[OrganizeResult]:
        indexed = await self._load_scope(payload, library)
        skipped = 0
        if payload.media_file_ids and not indexed:
            # 勾选快照里的行在本次运行前已消失 (自动整理的单文件范围是主要来源): 不是「空跑」, 记一次跳过.
            logger.warning("organize scope missing", library_id=library.id, media_file_ids=payload.media_file_ids)
            skipped += 1
        library_root = Path(library.path)
        media_extensions = frozenset(self._config.watcher.media_extensions) or MEDIA_EXTENSIONS
        scan = LibraryScan(
            trailer_pattern=library.trailer_pattern,
            blacklist_patterns=library.blacklist_patterns,
            min_file_size=library.min_file_size,
            media_extensions=media_extensions,
        )
        live: list[MediaFile] = []
        prune_total = len(indexed)
        if prune_total:
            await self.report_progress(0, prune_total, "prune")
            for i, mf in enumerate(indexed, start=1):
                if mf.id is not None:
                    mf_path = Path(mf.path)
                    missing = await existing_disk_path(mf_path, follow_symlinks=False) is None
                    if missing or not is_descendant(mf_path, library_root) or is_in_trash(mf_path):
                        await self._repo.delete_media_file(mf.id)
                    else:
                        kind = await _classify_indexed(mf_path, scan)
                        if kind is LibraryFileKind.UNWANTED or kind is LibraryFileKind.SKIP:
                            skipped += 1
                        else:
                            # MEDIA, 以及 classify 返回 None 的已索引行 (不在当前扩展名白名单,
                            # 例如库曾用 patterns 收过 .txt). 仍按索引落盘, 不按扫描 glob 再裁一次.
                            live.append(mf)
                await self.report_progress(i, prune_total, "prune")
        else:
            live = indexed

        organized = 0
        conflicted = 0
        conflicts: list[OrganizeConflict] = []
        failed = 0
        # 只有移动方式会移走源文件, 复制 / 硬链接 / 符号链接不腾空目录.
        prune_candidates: set[Path] = set()
        collect_prune = payload.prune_empty_dirs and library.move_mode is MoveMode.MOVE
        plans: dict[int, _RebuildPlan] = {}
        if payload.rebuild and live:
            before = {mf.id for mf in live}
            live, plans = await self._rebuild_items(live, library, payload)
            # 内容没变的条目整条跳过, 不重写任何产物; 计数并入 skipped 让用户看到「没动」.
            skipped += len(before - {mf.id for mf in live})
        total = len(live)
        if total == 0:
            await self.report_progress(1, 1, "done")
        else:
            await self.report_progress(0, total, "organize")
            for i, media_file in enumerate(live, start=1):
                path_str = media_file.path
                file_path = Path(path_str)
                metadata_id = media_file.metadata_id
                if metadata_id is None:
                    skipped += 1
                    await self.report_progress(i, total, file_path.name)
                    continue

                metadata = await self._repo.get_metadata(metadata_id)
                if metadata is None:
                    skipped += 1
                    await self.report_progress(i, total, file_path.name)
                    continue

                # 重建: 落盘之前先把该条目上一轮的产物 (含本轮同名的) 清掉, 目标位才会变 FREE.
                plan = plans.pop(metadata_id, None)
                if plan is not None:
                    source = await existing_disk_path(Path(media_file.path), follow_symlinks=False)
                    if source is None:
                        skipped += 1
                        await self.report_progress(i, total, file_path.name)
                        continue
                    removed = await _purge_item_dirs(
                        plan.directories,
                        plan.number,
                        set(plan.keep),
                        extrafanart_dirs=plan.extrafanart_dirs,
                        library_root=library_root,
                        preserve=set(plan.expected),
                    )
                    if removed:
                        logger.info(
                            "organize rebuild cleaned stale products",
                            number=plan.number,
                            expected=len(plan.expected),
                            removed=[str(item) for item in removed],
                        )

                write_nfo = library.write_nfo if payload.write_nfo is None else payload.write_nfo
                copy_resources = library.copy_resources if payload.copy_resources is None else payload.copy_resources
                try:
                    fop_result = await apply_file_operations(
                        self._repo,
                        media_file.id,
                        metadata,
                        self._config,
                        self._resource_store,
                        write_nfo=write_nfo,
                        copy_resources=copy_resources,
                        web_client=self._web_client,
                        safe_dirs=self._safe_dirs,
                        watermark_dir=self._watermark_dir,
                    )
                except Exception as exc:
                    # 单条目落盘失败只记一次失败, 不能让整库任务一起崩掉.
                    logger.exception("organize item failed", path=path_str, error=str(exc))
                    failed += 1
                    await self.report_progress(i, total, file_path.name)
                    continue
                if fop_result is None:
                    skipped += 1
                    await self.report_progress(i, total, file_path.name)
                    continue

                if fop_result.dest and media_file.id is not None:
                    await commit_organized_media_file(self._repo, media_file, fop_result.dest, library_root)
                match fop_result.outcome:
                    case PlaceOutcome.PLACED:
                        organized += 1
                        if collect_prune:
                            prune_candidates.update(ancestor_dirs(file_path, library_root=library_root))
                        # 字幕冲突的 path 是留在原处的那条字幕, 不是视频源路径.
                        conflicted += _record_conflicts(
                            conflicts,
                            [
                                OrganizeConflict(
                                    path=str(item.source),
                                    target=str(item.target),
                                    reason=OrganizeConflictReason.subtitle_target_exists,
                                )
                                for item in fop_result.subtitle_conflicts
                            ],
                        )
                    case PlaceOutcome.CONFLICT:
                        # 行不提交新路径, 留在源路径: 下次整理会再次报告, 直到用户自行处理.
                        target = fop_result.conflict_target
                        conflicted += _record_conflicts(
                            conflicts,
                            []
                            if target is None
                            else [
                                OrganizeConflict(
                                    path=path_str, target=str(target), reason=OrganizeConflictReason.target_exists
                                )
                            ],
                        )
                    case PlaceOutcome.FAILED:
                        logger.warning("organize failed", path=path_str, error=fop_result.error)
                        failed += 1
                await self.report_progress(i, total, file_path.name)
            await self.report_progress(total, total, "done")

        pruned_dirs = 0
        if prune_candidates:
            pruned_dirs = (await prune_empty_dirs(prune_candidates, library_root=library_root)).removed

        logger.info(
            "organize completed",
            path=payload.path,
            organized=organized,
            skipped=skipped,
            conflicted=conflicted,
            failed=failed,
            pruned_dirs=pruned_dirs,
        )

        return TaskResult(
            True,
            result=OrganizeResult(
                organized=organized,
                skipped=skipped,
                conflicted=conflicted,
                failed=failed,
                pruned_dirs=pruned_dirs,
                conflicts=conflicts,
            ),
        )


def _record_conflicts(conflicts: list[OrganizeConflict], entries: Sequence[OrganizeConflict]) -> int:
    """登记未处理的目标, 返回条数; 列表达到上限后只计数, 差额只留在任务日志里."""
    conflicts.extend(entries[: max(0, ORGANIZE_CONFLICT_LIMIT - len(conflicts))])
    return len(entries)


def _add_resource_ref(url: str, live_urls: set[str], live_hashes: set[str]) -> None:
    if not url:
        return
    url_hash = internal_url_hash(url)
    if url_hash is not None:
        live_hashes.add(url_hash)
    else:
        live_urls.add(url)


async def _collect_live_resource_refs(repo: Repository) -> tuple[set[str], set[str]]:
    live_urls: set[str] = set()
    live_hashes: set[str] = set()
    offset = 0
    page = 500
    # 收集 Metadata 仍引用的 locator / url_hash.
    while True:
        batch, _total = await repo.list_metadata(offset=offset, limit=page)
        if not batch:
            break
        for meta in batch:
            for u in (*meta.poster_urls, *meta.thumb_urls, *meta.trailer_urls):
                _add_resource_ref(u, live_urls, live_hashes)
            for site_urls in meta.extrafanart_urls.values():
                for u in site_urls:
                    _add_resource_ref(u, live_urls, live_hashes)
        offset += len(batch)
        if len(batch) < page:
            break

    offset = 0
    # 收集 Actor 头像 URL.
    while True:
        actors = await repo.list_actors(offset=offset, limit=page)
        if not actors:
            break
        for actor in actors:
            for u in actor.image_urls or []:
                _add_resource_ref(u, live_urls, live_hashes)
        offset += len(actors)
        if len(actors) < page:
            break
    return live_urls, live_hashes


@in_thread
def _missing_media_ids(rows: Sequence[tuple[int, str]]) -> list[int]:
    """不跟随符号链接判断存在."""
    return [media_id for media_id, path in rows if existing_disk_path_sync(path, follow_symlinks=False) is None]


class CleanupHandler(TaskHandler[CleanupPayload, CleanupResult]):
    """不因缺少 MediaFile 而删除 Metadata."""

    def __init__(self, repo: Repository, resource_store: ResourceStore):
        super().__init__(payload_t=CleanupPayload, result_t=CleanupResult)
        self._repo = repo
        self._resource_store = resource_store

    async def handle(self, payload: CleanupPayload) -> TaskResult[CleanupResult]:
        files_removed = 0
        resources_removed = 0

        # 删除磁盘上不存在的 MediaFile 记录.
        if payload.remove_missing_files:
            missing_ids: list[int] = []
            offset = 0
            page = 500
            while True:
                batch = await self._repo.list_media_files(limit=page, offset=offset)
                if not batch:
                    break
                missing_ids.extend(await _missing_media_ids([(mf.id, mf.path) for mf in batch if mf.id is not None]))
                offset += len(batch)
                if len(batch) < page:
                    break
            for media_id in missing_ids:
                await self._repo.delete_media_file(media_id)
                files_removed += 1

        # 删除不被 Metadata / Actor URL 引用的 Resource.
        if payload.remove_unreferenced_resources:
            live_urls, live_hashes = await _collect_live_resource_refs(self._repo)
            resources_removed = await self._resource_store.purge_unreferenced(live_urls, live_hashes)

        logger.info("cleanup completed", files_removed=files_removed, resources_removed=resources_removed)

        return TaskResult(True, result=CleanupResult(files_removed=files_removed, resources_removed=resources_removed))
