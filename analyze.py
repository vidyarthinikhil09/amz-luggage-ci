from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "analyze.py")
    raise SystemExit(subprocess.call([sys.executable, script_path, *sys.argv[1:]]))


if __name__ == "__main__":
    main()
