"""单向同步：P1-QuantFactor（实验工作树） -> _qf（git 仓 / 推送仓）。

背景
----
`E:/project/sj` 根目录**不是** git 仓。项目里存在两份 P1 工作树：

    E:/project/sj/P1-QuantFactor/   实验用，无 .git，跑训练/回测在这里
    E:/project/sj/_qf/              真 git 仓，用于推送

两者靠人工 cp 同步，已出现过漂移（`_qf` 曾缺 35/36/37/38 等 7 个脚本，
且 `src/` 与 `scripts/` 的改动容易只落一边）。本脚本把方向固化为
**单向 P1-QuantFactor -> _qf**，并在覆盖前打印 diff 摘要，避免静默丢改动。

原则
----
- **不删除目标侧文件**（只增/覆盖），因此 `_qf` 独有内容不会被误删。
- **不复制数据/模型/日志**（体积大且属产物，不入 git）。
- **覆盖前必须显示 diff**；`--check` 只报告不改动，退出码 1 表示存在差异（可用于门禁）。
- 同步后**不自动 commit**，由人确认 diff 后再提交（保持 git 历史的可审查性）。

用法
----
    python scripts/sync_to_qf.py --check        # 只报告差异（安全，建议先跑）
    python scripts/sync_to_qf.py --apply        # 实际同步
    python scripts/sync_to_qf.py --apply --dry  # 演练：打印将写的文件，不落盘
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]      # E:/project/sj/P1-QuantFactor
SRC = PROJ
DST = PROJ.parent / "_qf"

# 参与同步的目录/文件（相对项目根）。故意不含 data/ models/ logs/ signals/。
SYNC_TREES = ["src", "scripts", "tests", "config"]
SYNC_FILES = ["conftest.py", "README.md", ".gitignore"]

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".parquet", ".pt", ".onnx", ".csv"}


def _iter_files(base: Path, rel: str):
    """产出 rel 下所有应同步的文件（相对项目根的 PosixPath）。"""
    root = base / rel
    if not root.exists():
        return
    if root.is_file():
        yield Path(rel)
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.suffix in SKIP_SUFFIXES:
            continue
        yield p.relative_to(base)


def collect() -> list[Path]:
    rels: list[Path] = []
    for tree in SYNC_TREES:
        rels.extend(_iter_files(SRC, tree))
    for f in SYNC_FILES:
        if (SRC / f).is_file():
            rels.append(Path(f))
    # 去重并保持稳定顺序
    seen, out = set(), []
    for r in rels:
        key = r.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def collect_dst_only() -> list[Path]:
    """目标侧独有的受管文件（源侧不存在）。

    为什么需要它：`collect()` 只从**源侧**枚举，因此目标侧多出来的文件永远看不见。
    这曾导致一个真实漏检——在 `_qf/src/` 手写一个源侧没有的文件，
    `--check` 与门禁都报「一致」。单向同步的语义是「目标侧的内容应等于源侧」，
    所以目标侧独有文件同样是漂移，必须报出来。
    注意这些文件**不会被自动删除**（只报告，交人判断是补到源侧还是删掉）。
    """
    if not DST.is_dir():
        return []
    src_set = {r.as_posix() for r in collect()}
    out, seen = [], set()
    for tree in SYNC_TREES:
        for rel in _iter_files(DST, tree):
            key = rel.as_posix()
            if key in src_set or key in seen:
                continue
            seen.add(key)
            out.append(rel)
    return out


def compare() -> tuple[list[Path], list[Path], list[Path]]:
    """返回 (新增, 内容不同, 已一致)。已一致不含目标侧独有文件。"""
    new, changed, same = [], [], []
    for rel in collect():
        s, d = SRC / rel, DST / rel
        if not d.exists():
            new.append(rel)
        elif not filecmp.cmp(s, d, shallow=False):
            changed.append(rel)
        else:
            same.append(rel)
    return new, changed, same


def _print_diff(rel: Path, limit: int = 12) -> None:
    import difflib
    try:
        a = (DST / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        b = (SRC / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:                                  # pragma: no cover
        print(f"      (diff 读取失败: {e})")
        return
    diff = list(difflib.unified_diff(a, b, fromfile=f"_qf/{rel}", tofile=f"P1/{rel}", lineterm=""))
    if not diff:
        return
    for line in diff[:limit]:
        print(f"      {line}")
    if len(diff) > limit:
        print(f"      ... 另有 {len(diff) - limit} 行差异")


def main() -> int:
    ap = argparse.ArgumentParser(description="P1-QuantFactor -> _qf 单向同步")
    ap.add_argument("--check", action="store_true", help="只报告差异，不写入；有差异退出码 1")
    ap.add_argument("--apply", action="store_true", help="实际同步")
    ap.add_argument("--dry", action="store_true", help="配合 --apply：只打印不落盘")
    ap.add_argument("--max-diff", type=int, default=6, help="最多展示几个文件的详细 diff")
    args = ap.parse_args()

    if not args.check and not args.apply:
        ap.error("请指定 --check 或 --apply")

    if not DST.is_dir():
        print(f"[FAIL] 目标仓不存在: {DST}")
        return 2

    new, changed, same = compare()
    dst_only = collect_dst_only()
    print(f"源  : {SRC}")
    print(f"目标: {DST}")
    print(f"应同步文件 {len(new) + len(changed) + len(same)} 个 | "
          f"新增 {len(new)} | 内容不同 {len(changed)} | 已一致 {len(same)} | "
          f"目标侧独有 {len(dst_only)}")
    print("-" * 62)

    for rel in new:
        print(f"  [新增] {rel.as_posix()}")
    shown = 0
    for rel in changed:
        print(f"  [覆盖] {rel.as_posix()}")
        if shown < args.max_diff:
            _print_diff(rel)
            shown += 1
    if len(changed) > args.max_diff:
        print(f"  ... 另有 {len(changed) - args.max_diff} 个文件有差异未展开")
    for rel in dst_only:
        print(f"  [目标侧独有·需人工判断] {rel.as_posix()}")

    todo = len(new) + len(changed)
    if todo == 0 and not dst_only:
        print("\n两侧已一致，无需同步。")
        return 0

    if args.check:
        parts = []
        if todo:
            parts.append(f"{todo} 处待同步")
        if dst_only:
            parts.append(f"{len(dst_only)} 个目标侧独有文件（本工具不删，需人工判断）")
        print(f"\n[CHECK] " + "；".join(parts) + "。未修改任何文件。")
        return 1

    if args.dry:
        print(f"\n[DRY] 将写入 {todo} 个文件（本次未落盘）。")
        if dst_only:
            print(f"      {len(dst_only)} 个目标侧独有文件不会被删除，需人工判断。")
        return 0

    written = 0
    for rel in new + changed:
        s, d = SRC / rel, DST / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        written += 1
    print(f"\n[OK] 已同步 {written} 个文件到 _qf。")
    if dst_only:
        print(f"     ⚠️ 目标侧另有 {len(dst_only)} 个源侧不存在的受管文件，本工具**不删除**：")
        for rel in dst_only:
            print(f"        - {rel.as_posix()}")
        print("        请判断：应保留（则补到 P1-QuantFactor）还是删除（则手工 rm）。")
    print("     下一步：cd _qf && git status && git diff，确认后自行 commit。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
