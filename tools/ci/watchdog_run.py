"""Run `python -m cbtv.run_tracker ARGS...` in-process with a hang watchdog (C014, macOS diagnosis).

If the tracker runs longer than CBTV_WATCHDOG_S seconds (default 300), every thread's stack is dumped to stderr
and the process exits (faulthandler.dump_traceback_later). Usage: python tools/ci/watchdog_run.py <tracker args>"""
import faulthandler
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so cbtv imports from the checkout

limit = float(os.environ.get("CBTV_WATCHDOG_S", "300"))
faulthandler.dump_traceback_later(limit, exit=True)
sys.argv = ["cbtv.run_tracker"] + sys.argv[1:]
runpy.run_module("cbtv.run_tracker", run_name="__main__")
