"""Run docs/tutorial_fixture.md as written: execute every shell line of its fenced blocks that starts with
pip / cbtv / cmp, in order, from the repo root; then check each line of the expected summary.txt block appears in
tutorial_out/summary.txt. Exit 0 only if every command exits 0 and the summary matches.
Usage: python tools/docs/run_tutorial.py [--dry-run]"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "tutorial_fixture.md"
COMMANDS = ("pip ", "cbtv ", "cmp ")


def blocks(text):
    return [b.strip("\n").splitlines() for b in re.findall(r"^```[^\n]*\n(.*?)^```", text, re.S | re.M)]


def main():
    dry = "--dry-run" in sys.argv
    bl = blocks(DOC.read_text(encoding="utf-8"))
    cmds = [ln.strip() for b in bl for ln in b if ln.strip().startswith(COMMANDS)]
    summary = next(b for b in bl if b and b[0].startswith("frames processed"))
    fails = 0
    for c in cmds:
        print("$ " + c, flush=True)
        if not dry:
            rc = subprocess.call(c, shell=True, cwd=str(ROOT))
            print("  -> exit %d" % rc, flush=True)
            fails += rc != 0
    if not dry:
        got = {" ".join(ln.split()) for ln in (ROOT / "tutorial_out" / "summary.txt").read_text().splitlines()}
        for ln in summary:
            ok = " ".join(ln.split()) in got
            print("%s summary: %s" % ("ok  " if ok else "MISS", ln))
            fails += not ok
    print("failed checks: %d" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
