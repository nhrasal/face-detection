#!/usr/bin/env python3
"""Fetch public-domain test portraits from Wikimedia Commons.

The model tier of the test suite needs real faces, and specifically MULTIPLE
photos of the SAME person — that is the only way to test an identity decision.
Synthetic images cannot do it.

These are NASA photographs: works of US federal employees, hence public domain.
The licence of every file is verified against the Commons API before download;
anything not explicitly public domain is refused rather than downloaded and
sorted out later.

Downloads land in backend/tests/assets/, which is gitignored. Face photos must
never enter git history.

    python backend/scripts/fetch_test_faces.py
    python backend/scripts/fetch_test_faces.py --verify   # check what is on disk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "face-detection-dev/0.1 (test asset fetch; contact: local development)"
THUMB_WIDTH = 900

BACKEND = Path(__file__).resolve().parent.parent
ASSET_DIR = BACKEND / "tests" / "assets"

# dest filename -> Commons "File:" title.
#
# Five identities with two or three portraits each. Multiple identities matter
# as much as multiple photos: genuine pairs alone only prove the model says
# "same", and a model that says "same" to everything would pass. Impostor pairs
# across identities are what make the assertion two-sided.
#
# Every file here was verified to yield exactly ONE detected face. Candidates
# that did not are recorded below so nobody re-adds them.
ASSETS: dict[str, str] = {
    "faces/jemison_1.jpg": "Mae Jemison - Official portrait of 1987 astronaut candidate.jpg",
    "faces/jemison_2.jpg": "Mae-jemison.jpg",
    "faces/jemison_3.jpg": "Mae Carol Jemison (cropped).jpg",
    "faces/ride_1.jpg": "Sally Ride (1984).jpg",
    "faces/ride_2.jpg": "S83-35763 (cropped).jpg",
    "faces/bluford_1.jpg": "Guion Bluford - 1978 (close up cropped).jpg",
    "faces/bluford_2.jpg": "Guion Bluford cropped.jpg",
    "faces/collins_1.jpg": "Eileen Collins, early NASA portrait.jpg",
    "faces/collins_2.jpg": "Commander Eileen Collins - GPN-2000-001177.jpg",
    "faces/aldrin_1.jpg": "Buzz Aldrin.jpg",
    "faces/aldrin_2.jpg": "Buzz Aldrin black and white dress uniform photo portrait.jpg",
    # Negative cases for the pipeline tier.
    "group.jpg": "STS-47 crew.jpg",  # 7 faces -> MULTIPLE_FACES
    "no_face.jpg": "The Earth seen from Apollo 17.jpg",  # 0 faces -> NO_FACE
}

# Rejected candidates, kept as a record so they are not tried again:
#   "Aldrin Gemini XII 3.jpg"        0 faces — pressure helmet and visor
#   "Simulator - Ride, Sally K..jpg" 2 faces — a second person in frame
#   "Apollo 11 Buzz Aldrin.JPG"      CC BY-SA 3.0, not public domain


def _get(url: str, *, attempts: int = 5) -> bytes:
    """Fetch with polite throttling and backoff.

    Commons rate-limits anonymous bulk downloads and answers 429. Hammering it
    is both rude and ineffective, so pause between requests and back off when
    asked to.
    """
    delay = 2.0
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts:
                raise
            wait = float(exc.headers.get("Retry-After") or delay)
            print(f"        rate limited, waiting {wait:.0f}s (attempt {attempt}/{attempts})")
            time.sleep(wait)
            delay *= 2
    raise RuntimeError("unreachable")


def resolve(titles: list[str]) -> dict[str, dict[str, str]]:
    """Look up thumbnail URL and licence for each Commons file title."""
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": str(THUMB_WIDTH),
            "titles": "|".join(f"File:{t}" for t in titles),
        }
    )
    payload = json.loads(_get(f"{API}?{query}"))
    out: dict[str, dict[str, str]] = {}
    for page in payload["query"]["pages"].values():
        title = page["title"].removeprefix("File:")
        info = page.get("imageinfo")
        if not info:
            continue
        meta = info[0].get("extmetadata", {})
        out[title] = {
            "url": info[0].get("thumburl") or info[0]["url"],
            "licence": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "credit": meta.get("Artist", {}).get("value", "NASA"),
            "descriptionurl": info[0].get("descriptionurl", ""),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check disk, download nothing")
    args = parser.parse_args()

    if args.verify:
        missing = [d for d in ASSETS if not (ASSET_DIR / d).exists()]
        for dest in ASSETS:
            path = ASSET_DIR / dest
            mark = "ok  " if path.exists() else "MISS"
            size = f"{path.stat().st_size:>9,} bytes" if path.exists() else ""
            print(f"  {mark} {dest:<24} {size}")
        if missing:
            print(f"\n{len(missing)} missing — run without --verify", file=sys.stderr)
            return 1
        print(f"\nall {len(ASSETS)} assets present")
        return 0

    print(f"resolving {len(ASSETS)} files via the Commons API…")
    resolved = resolve(list(ASSETS.values()))

    attribution = [
        "Test assets fetched from Wikimedia Commons by scripts/fetch_test_faces.py.",
        "All files are NASA photographs and public domain (works of US federal employees).",
        "This directory is gitignored: face photos must not enter git history.",
        "",
    ]

    for dest, title in ASSETS.items():
        meta = resolved.get(title)
        if meta is None:
            print(f"  FAIL  {dest}: '{title}' not found on Commons", file=sys.stderr)
            return 1

        # Licence gate. Refuse anything that is not explicitly public domain
        # rather than downloading it and sorting out attribution later.
        if "public domain" not in meta["licence"].lower():
            print(
                f"  FAIL  {dest}: licence is '{meta['licence']}', not public domain",
                file=sys.stderr,
            )
            return 1

        path = ASSET_DIR / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"  have  {dest}")
        else:
            data = _get(meta["url"])
            path.write_bytes(data)
            time.sleep(1.5)  # be a good citizen between downloads
            print(f"  got   {dest:<24} {len(data):>9,} bytes  [{meta['licence']}]")

        attribution.append(f"{dest}\n    {title}\n    {meta['licence']} — {meta['descriptionurl']}")

    (ASSET_DIR / "ATTRIBUTION.txt").write_text("\n".join(attribution) + "\n")
    print(f"\nwrote {ASSET_DIR / 'ATTRIBUTION.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
