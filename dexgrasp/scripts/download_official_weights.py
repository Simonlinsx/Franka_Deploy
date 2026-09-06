#!/usr/bin/env python3
"""Download the minimal public AnyDexGrasp inference checkpoints.

The file IDs and relative paths come from the public Google Drive folder linked
by the upstream README.  Training datasets and labels are intentionally omitted.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time


FILES = {
    "logs/model/checkpoint.tar.18": "1r65_VPY_326sn0xANz2I8ZiL9LuJ5afY",
    "logs/model/inspire_model/obj140/480/1/0.9_1.0_0.08_0.1481_39.pth": "1duvg8SAU04enFpK2cuxPmjXen3kvIPHO",
    "logs/model/inspire_model/obj140/480/2/0.9_1.0_0.18_0.3051_38.pth": "1ivdbYFjPjGHYGqAmJR_2qQvfPgsqRVBk",
    "logs/model/inspire_model/obj140/480/3/0.9_1.0_0.22_0.3607_62.pth": "1rmKx5j1zQA7l4Z_1yVEajchQYwT0Q-YA",
    "logs/model/inspire_model/obj140/480/4/0.9_1.0_0.08_0.1481_17.pth": "1bOsTxC6B85KtPrrMoX18dbDspNRSfEfo",
    "logs/model/inspire_model/obj140/480/5/0.9_1.0_0.08_0.1481_36.pth": "1OlkwVkwoL9lVBfKos9_C0L-tlrOjGtVX",
    "logs/model/inspire_model/obj140/480/6/0.9_1.0_0.18_0.3051_76.pth": "1TYU_hucug6R_c0vfPtTiUPv8nnBRq5zk",
    "logs/model/inspire_model/obj140/480/7/0.9_0.875_0.14_0.2414_34.pth": "1KrHApzJYrQlXtgitdR_Uaw4yQuldltIq",
    "logs/model/inspire_model/obj140/480/8/0.9_0.6_0.06_0.1091_40.pth": "1ZXBgupOfhis8mhqFJukSAnma0FFA-TgY",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "weights",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()
    for relative, file_id in FILES.items():
        print(f"{file_id}  {relative}")
    if args.list:
        return 0

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required only for this downloader; install it with "
            "`python -m pip install gdown`"
        ) from exc

    for relative, file_id in FILES.items():
        output = args.output_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.stat().st_size > 0 and not args.force:
            print(f"[skip] {output} ({output.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[download] {relative}")
        result = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                result = gdown.download(
                    id=file_id, output=str(output), quiet=False, resume=True
                )
                break
            except Exception as exc:
                if attempt >= max(1, args.retries):
                    raise
                delay = min(10, attempt * 2)
                print(
                    f"[retry {attempt}/{args.retries}] {type(exc).__name__}: "
                    f"{exc}; resuming in {delay}s"
                )
                time.sleep(delay)
        if result is None or not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"download failed: {relative}")
        print(
            f"[ok] {output} size={output.stat().st_size / 1e6:.1f}MB "
            f"sha256={sha256(output)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
