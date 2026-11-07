"""
Mengubah batas GADM 4.1 level 2 (GeoJSON) menjadi CSV dengan kolom geometri WKT, supaya dapat
diunggah sebagai aset tabel Earth Engine. Unggahan tabel di Code Editor hanya menerima shapefile
atau CSV, bukan GeoJSON (developers.google.com/earth-engine/guides/table_upload, dicek 25
September 2026); shapefile perlu pustaka yang tidak terpasang, CSV tidak.

Kolom yang disimpan: GID_2 (kunci penggabung panel), NAME_1, NAME_2, TYPE_2, WKT.

Pakai:
    python gadm_ke_csv_wkt.py
Keluaran: DL_FIRE_SV-C2_792597/gadm41_IDN_2_wkt.csv
Saat mengunggah: Assets -> New -> CSV file, lalu isi "Geometry column" dengan WKT.
"""

import csv
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SUMBER = os.path.join(BASE, "DL_FIRE_SV-C2_792597", "gadm41_IDN_2.json")
KELUAR = os.path.join(BASE, "DL_FIRE_SV-C2_792597", "gadm41_IDN_2_wkt.csv")


def cincin(r):
    return "(" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in r) + ")"


def ke_wkt(g):
    if g["type"] == "Polygon":
        return "POLYGON (" + ", ".join(cincin(r) for r in g["coordinates"]) + ")"
    if g["type"] == "MultiPolygon":
        return "MULTIPOLYGON (" + ", ".join(
            "(" + ", ".join(cincin(r) for r in p) + ")" for p in g["coordinates"]) + ")"
    raise ValueError(g["type"])


g = json.load(open(SUMBER, encoding="utf-8"))
csv.field_size_limit(10 ** 9)
with open(KELUAR, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["GID_2", "NAME_1", "NAME_2", "TYPE_2", "WKT"])
    for ft in g["features"]:
        p = ft["properties"]
        w.writerow([p["GID_2"], p["NAME_1"], p["NAME_2"], p["TYPE_2"], ke_wkt(ft["geometry"])])
print(f"{len(g['features'])} satuan -> {KELUAR} ({os.path.getsize(KELUAR) / 1e6:.1f} MB)")
