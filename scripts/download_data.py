"""Download raw historical data: NASA OMNI2 hourly and GFZ Potsdam Kp.

Usage:
    python scripts/download_data.py           # skip files that already exist
    python scripts/download_data.py --force   # re-download everything
"""

import argparse
from pathlib import Path

import httpx

from aurora.config import GFZ_KP_URL, OMNI2_URL, RAW_DIR

SOURCES = {
    "omni2_all_years.dat": OMNI2_URL,
    "Kp_ap_Ap_SN_F107_since_1932.txt": GFZ_KP_URL,
}


def download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"skip  {dest.name} (exists, {dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)  # atomic: never leave a half-written file under the real name
    print(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    for name, url in SOURCES.items():
        download(url, RAW_DIR / name, force=args.force)


if __name__ == "__main__":
    main()
