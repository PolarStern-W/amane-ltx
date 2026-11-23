#!/usr/bin/env python3
"""把本地迁移重新接到上游最新的 head 上, 让 alembic 只留一个 head.

`ltx` 分支带一条本地数据迁移 (文件名带 -LTX 的行重算为 LTX 标记). 上游每加一个迁移,
迁移图就多出一条 head, `alembic upgrade head` 会因为多 head 直接失败.
同步作业在合并上游之后调用本脚本: 把本地迁移的 down_revision 改指上游 head.

用法: python3 .github/scripts/realign-migration-heads.py [仓库根]
退出码 0 = 已是单 head; 1 = 修不了 (需要人工看).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VERSIONS_DIR = Path("src/amane/db/migrations/versions")
LOCAL_MARKER = "split_ltx_media_file_mosaic_from_cracked"
REVISION_RE = re.compile(r'^revision(?::\s*str)?\s*=\s*"([0-9a-f]+)"', re.MULTILINE)
DOWN_REVISION_RE = re.compile(r"^(down_revision(?::[^=]+)?\s*=\s*)(.+)$", re.MULTILINE)
PARENT_RE = re.compile(r'"([0-9a-f]+)"')


def read_graph(versions_dir: Path) -> dict[str, tuple[Path, str, list[str]]]:
    graph: dict[str, tuple[Path, str, list[str]]] = {}
    for path in sorted(versions_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        revision = REVISION_RE.search(text)
        if revision is None:
            continue
        down_revision = DOWN_REVISION_RE.search(text)
        parents = PARENT_RE.findall(down_revision.group(2)) if down_revision is not None else []
        graph[revision.group(1)] = (path, text, parents)
    return graph


def heads(graph: dict[str, tuple[Path, str, list[str]]]) -> list[str]:
    parents = {parent for _, _, children in graph.values() for parent in children}
    return sorted(revision for revision in graph if revision not in parents)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    versions_dir = root / VERSIONS_DIR
    graph = read_graph(versions_dir)
    current = heads(graph)
    print(f"migration heads: {current}")
    if len(current) <= 1:
        print("single head, nothing to do")
        return 0

    local = [revision for revision, (path, _, _) in graph.items() if LOCAL_MARKER in path.name]
    if len(local) != 1:
        print(f"error: 找不到本地迁移 {LOCAL_MARKER}", file=sys.stderr)
        return 1
    local_revision = local[0]
    if local_revision not in current:
        print("error: 本地迁移不是 head, 上游改动过迁移历史", file=sys.stderr)
        return 1

    upstream = [revision for revision in current if revision != local_revision]
    if len(upstream) != 1:
        print(f"error: 期望一条上游 head, 实际 {upstream}", file=sys.stderr)
        return 1

    target = upstream[0]
    path, text, _ = graph[local_revision]
    updated = DOWN_REVISION_RE.sub(lambda match: f'{match.group(1)}"{target}"', text, count=1)
    updated = re.sub(r"(?m)^Revises: .*$", f"Revises: {target}", updated, count=1)
    path.write_text(updated, encoding="utf-8")
    print(f"repointed {local_revision} onto {target}")

    after = heads(read_graph(versions_dir))
    print(f"migration heads after: {after}")
    return 0 if len(after) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
