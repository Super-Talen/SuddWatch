"""
SuddWatch - build the static assets for the public web page (api/static/).

1. data/counties.geojson: the 79 county outlines (OCHA COD-AB admin 2), simplified to ~300 m and rounded,
   so the map loads quickly on slow connections.
2. img/*.jpg + img/credits.json: openly licensed flood photographs and satellite images from Wikimedia
   Commons, resized for the web. Licenses were checked through the Commons API on 2026-11-24.
   The originals are downloaded by hand (Wikimedia blocks anonymous scripted downloads) into
   data/raw/web_photos/, saved under either the key name (e.g. maban_flooding.jpg) or the Commons file name.

Usage: .venv\\Scripts\\python.exe -m src.build_web_assets
"""
import json
import re
from pathlib import Path

import geopandas as gpd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "api/static"
PHOTO_SRC = ROOT / "data/raw/web_photos"
COMMONS = "https://commons.wikimedia.org/wiki/File:"
CC_BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"

PHOTOS = [
    {"key": "jonglei_children", "file": "Children_in_a_Jonglei_state_of_South_Sudan_draining_water_due_to_flooding.jpg",
     "title": "Children draining floodwater, Jonglei State (2023)", "author": "Aropmathewdaniel",
     "license": "CC BY-SA 4.0", "license_url": CC_BY_SA},
    {"key": "bentiu_before_after", "file": "Floods_in_South_Sudan_(Copernicus).jpg",
     "title": "Bentiu, Unity State: 21 Nov 2020 and 26 Nov 2021 (Sentinel-2, false colour)",
     "author": "European Union, Copernicus Sentinel-2 imagery", "license": "Attribution", "license_url": None},
    {"key": "bentiu_from_space", "file": "Earth_from_Space-_Bentiu,_South_Sudan_ESA25136828.jpg",
     "title": "Bentiu and the surrounding wetlands from space (Sentinel-2, false colour, published 2023)",
     "author": "ESA, contains modified Copernicus Sentinel data", "license": "Attribution", "license_url": None},
    {"key": "maban_flooding", "file": "Flooding_in_Maban_county.jpg",
     "title": "Flooding in Maban County, Upper Nile (September 2022)", "author": "Data Bettuel",
     "license": "CC BY-SA 4.0", "license_url": CC_BY_SA},
    {"key": "health_facility", "file": "Health_Facility_Flooded.jpg",
     "title": "Health facility abandoned after flooding, Lakes State (2021)", "author": "Odokonyeromari",
     "license": "CC BY-SA 4.0", "license_url": CC_BY_SA},
]


def counties():
    g = gpd.read_file(ROOT / "data/raw/ssd_admin_boundaries/ssd_admin2.geojson")
    g = g[["adm2_pcode", "adm2_name", "adm1_name", "geometry"]].to_crs(4326)
    g["geometry"] = g.geometry.simplify(0.003, preserve_topology=True)
    out = STATIC / "data/counties.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    gj = json.loads(g.to_json(drop_id=True))

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 4), round(c[1], 4)]
    for f in gj["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    out.write_text(json.dumps(gj, separators=(",", ":")), encoding="utf-8")
    print(f"counties.geojson: {len(gj['features'])} counties, {out.stat().st_size / 1024:.0f} KB")


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", Path(name).stem.lower())


def _find(p):
    """The hand-downloaded original: key name or Commons file name (spaces/underscores/case ignored)."""
    wanted = {_norm(p["key"]), _norm(p["file"])}
    for f in PHOTO_SRC.glob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff") and _norm(f.name) in wanted:
            return f
    return None


def photos(max_w=1600):
    out = STATIC / "img"
    out.mkdir(parents=True, exist_ok=True)
    credits, missing = [], []
    for p in PHOTOS:
        src = _find(p)
        if src is None:
            missing.append(p["file"])
            continue
        im = Image.open(src).convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        dst = out / f"{p['key']}.jpg"
        im.save(dst, "JPEG", quality=80, optimize=True, progressive=True)
        credits.append({**p, "src": f"/static/img/{p['key']}.jpg", "source_url": COMMONS + p["file"],
                        "license_url": p["license_url"] or COMMONS + p["file"], "width": im.width, "height": im.height})
        print(f"{dst.name}: {im.width}x{im.height}, {dst.stat().st_size / 1024:.0f} KB  (from {src.name})")
    (out / "credits.json").write_text(json.dumps(credits, indent=2), encoding="utf-8")
    if missing:
        print(f"\nNot found in {PHOTO_SRC.relative_to(ROOT)} (the page hides these until they are added):")
        for m in missing:
            print("  ", COMMONS + m)


if __name__ == "__main__":
    PHOTO_SRC.mkdir(parents=True, exist_ok=True)
    counties()
    photos()
