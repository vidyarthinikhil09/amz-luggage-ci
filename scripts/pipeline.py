from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    # Simple orchestrator; edit brand list here if needed.
    brands = ["Safari", "Skybags", "American Tourister", "VIP", "Aristocrat", "Nasher Miles"]

    run([sys.executable, os.path.join("scripts", "scrape_products.py"), "--brands", *brands, "--max-products-per-brand", "12"])
    run([sys.executable, os.path.join("scripts", "scrape_reviews.py"), "--max-reviews-per-asin", "40", "--max-reviews-per-brand", "120"])
    run([sys.executable, os.path.join("scripts", "analyze.py")])


if __name__ == "__main__":
    main()
