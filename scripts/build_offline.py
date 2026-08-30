"""Build the self-contained offline copy of the 3D digital twin.

Downloads three.js + OrbitControls, inlines them into frontend/tyre3d.html and
writes frontend/tyre3d.offline.html. The result makes no network requests, so
it can be presented from a USB stick or a room with no wifi.

    python scripts/build_offline.py
"""

from __future__ import annotations

import pathlib
import re
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "frontend" / "tyre3d.html"
DST = ROOT / "frontend" / "tyre3d.offline.html"
VENDOR = ROOT / "frontend" / "vendor"

ASSETS = {
    "three.min.js": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "OrbitControls.js": "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
}


def fetch(name: str, url: str) -> str:
    VENDOR.mkdir(parents=True, exist_ok=True)
    path = VENDOR / name
    if not path.exists():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding="utf-8")


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    for name, url in ASSETS.items():
        tag = f'<script src="{url}"></script>'
        if tag not in html:
            raise SystemExit(f"script tag for {name} not found in {SRC.name}")
        html = html.replace(
            tag, f"<script>/* {name} r128 — MIT — vendored for offline use */\n{fetch(name, url)}\n</script>", 1
        )

    # Google Fonts cannot be reached offline; fall back to the stacks already
    # declared in --font-h / --font-m rather than leaving a hanging request.
    html = re.sub(r'<link rel="preconnect" href="https://fonts\.(googleapis|gstatic)\.com"[^>]*>\s*', "", html)
    html = re.sub(r'<link href="https://fonts\.googleapis\.com/css2[^>]*>\s*', "", html)

    html = html.replace("3D Digital Twin</title>", "3D Digital Twin (offline)</title>", 1)
    html = html.replace("Digital Twin — v6.0.0", "Digital Twin — v6.0.0 · offline", 1)

    DST.write_text(html, encoding="utf-8")

    leftover = [u for u in re.findall(r'(?:src|href)="(https?://[^"]+)"', html) if "mendeley" not in u]
    print(f"wrote {DST.relative_to(ROOT)}  {DST.stat().st_size / 1024:.1f} KB")
    print(f"load-time external requests: {len(leftover)}")
    if leftover:
        raise SystemExit("offline build still references: " + ", ".join(leftover))


if __name__ == "__main__":
    main()
