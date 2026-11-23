"""Command line: python check_house.py photo.jpg [more.jpg ...]

Prints ACCEPT if the photo shows a house with a door or gate visible, otherwise REJECT.
"""

import argparse
import sys

from checker import HouseDoorChecker


def main() -> int:
    ap = argparse.ArgumentParser(description="Accept photos of a house with a visible door or gate.")
    ap.add_argument("images", nargs="+", help="image file(s) to check")
    ap.add_argument("--threshold", type=float, default=None, help="0..1, higher = stricter (default 0.5)")
    ap.add_argument("--verbose", action="store_true", help="also print the confidence score")
    args = ap.parse_args()

    checker = HouseDoorChecker(threshold=args.threshold)
    for path in args.images:
        try:
            r = checker.check(path)
        except Exception as e:  # unreadable file etc.
            print(f"{path}: ERROR ({e})", file=sys.stderr)
            continue
        prefix = f"{path}: " if len(args.images) > 1 else ""
        extra = f"  (score {r.probability:.2f})" if args.verbose else ""
        print(f"{prefix}{r.label}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
