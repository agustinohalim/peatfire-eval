"""
Komponen Fire Weather Index System (DC, DMC, FFMC, BUI, FWI) bulanan per kabupaten dari GFWED
(telaah editor Natural Hazards, 22 Oktober 2026, X2: salinan GFWED di Earth Engine hanya berpita FWI).

Sumber: GFWED v2.0, berkas rata-rata bulanan NASA NCCS
  https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.MERRA2/Default/<PRESIP>/YYYY/
  FWI.<PRESIP>.Monthly.Default.YYYYMM.nc   (MERRA-2 0,5 x 0,625 derajat; Field et al. 2015)
Berkas diunduh ke DL_FIRE_NASIONAL/gfwed_bulanan/ (tidak masuk repo). PRESIP dipilih dengan
--presip; yang dipakai naskah ditentukan dengan mencocokkan FWI hasil skrip ini dengan
fwi_bulanan_kabupaten_2012_2025_v2.csv (salinan Earth Engine) — lihat --cocokkan.

Agregasi meniru Skrip_FWI_Earth_Engine.js: titik sampel 0,05 derajat di dalam poligon GADM 4.1
level 2, tiap titik mengambil sel GFWED terdekat yang bernilai (sel laut/tanpa vegetasi kosong),
lalu dirata-rata. Kabupaten tanpa titik sampel (lebih kecil dari 0,05 derajat) memakai pusatnya.

Pakai:
  python gabung_gfwed_komponen.py --unduh MERRA2           (168 berkas, bisa berjam-jam)
  python gabung_gfwed_komponen.py --cocokkan                (bandingkan FWI dengan salinan EE, 2015-09)
  python gabung_gfwed_komponen.py --presip MERRA2           (tulis DL_FIRE_NASIONAL/gfwed_komponen_<PRESIP>.csv)
"""
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.io import netcdf_file
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
NC = os.path.join(D, "gfwed_bulanan")
GADM = os.path.join(BASE, "DL_FIRE_SV-C2_792597", "gadm41_IDN_2.json")
URL = "https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.MERRA2/Default"
KOMP = ["DC", "DMC", "FFMC", "BUI", "FWI"]
BULAN = [f"{t}{b:02d}" for t in range(2012, 2026) for b in range(1, 13)]
LANGKAH = 0.05


def berkas(presip, ym):
    return os.path.join(NC, f"FWI.{presip}.Monthly.Default.{ym}.nc")


def unduh(presip):
    os.makedirs(NC, exist_ok=True)
    for ym in BULAN:
        f = berkas(presip, ym)
        if os.path.exists(f) and os.path.getsize(f) > 1_000_000:
            continue
        subprocess.run(["curl", "-sL", "-m", "600", "-o", f, f"{URL}/{presip}/{ym[:4]}/{os.path.basename(f)}"], check=True)
        print(ym, os.path.getsize(f), flush=True)


def titik_kabupaten():
    gj = json.load(open(GADM, encoding="utf-8"))
    out = {}
    for ft in gj["features"]:
        g, geom = ft["properties"]["GID_2"], ft["geometry"]
        polis = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        pts = []
        semua = np.concatenate([np.asarray(p[0]) for p in polis])
        for p in polis:
            cincin = np.asarray(p[0])
            x0, y0 = cincin.min(0); x1, y1 = cincin.max(0)
            gx, gy = np.meshgrid(np.arange(x0 + LANGKAH / 2, x1, LANGKAH), np.arange(y0 + LANGKAH / 2, y1, LANGKAH))
            q = np.c_[gx.ravel(), gy.ravel()]
            if len(q):
                masuk = Path(cincin).contains_points(q)
                for lubang in p[1:]:
                    masuk &= ~Path(np.asarray(lubang)).contains_points(q)
                pts.append(q[masuk])
        pts = np.concatenate(pts) if pts else np.empty((0, 2))
        out[g] = pts if len(pts) else semua.mean(0, keepdims=True)
    return out


def baca(f, presip):
    n = netcdf_file(f, "r", mmap=False)
    lat, lon = n.variables["lat"][:].copy(), n.variables["lon"][:].copy()
    lon = np.where(lon > 180, lon - 360, lon)
    data = {}
    for k in KOMP:
        v = n.variables[f"{presip}_{k}"]
        a = v[0].astype(float)
        fv = getattr(v, "_FillValue", None) or getattr(v, "missing_value", None)
        if fv is not None:
            a[a == fv] = np.nan
        a[(a < -1) | (a > 1e5)] = np.nan
        data[k] = a
    n.close()
    return lat, lon, data


def agregasi(presip, bulan=BULAN):
    titik = titik_kabupaten()
    baris = []
    pohon = None
    for ym in bulan:
        lat, lon, data = baca(berkas(presip, ym), presip)
        LO, LA = np.meshgrid(lon, lat)
        sah = np.isfinite(data["FWI"]) & (LO > 90) & (LO < 145) & (LA > -15) & (LA < 10)
        if pohon is None or not np.array_equal(sah, sah0):
            pohon, sah0 = cKDTree(np.c_[LO[sah], LA[sah]]), sah
            idx = {g: pohon.query(p)[1] for g, p in titik.items()}
        for g, j in idx.items():
            r = {"GID_2": g, "bulan": f"{ym[:4]}-{ym[4:]}", "titik": len(titik[g])}
            for k in KOMP:
                r[k.lower()] = float(np.nanmean(data[k][sah][j]))
            baris.append(r)
        print(ym, flush=True)
    return pd.DataFrame(baris)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unduh")
    ap.add_argument("--cocokkan", action="store_true")
    ap.add_argument("--presip")
    a = ap.parse_args()
    if a.unduh:
        unduh(a.unduh)
    if a.cocokkan:
        ee = pd.read_csv(os.path.join(D, "fwi_bulanan_kabupaten_2012_2025_v2.csv"))
        ee = ee[ee.bulan == "2015-09"].set_index("GID_2").fwi
        for presip in ("MERRA2", "MERRA2.CORRECTED"):
            k = agregasi(presip, ["201509"]).set_index("GID_2").fwi
            j = ee.dropna().index.intersection(k.dropna().index)
            print(f"{presip}: n = {len(j)} (kosong skrip {k.isna().sum()}), r = {np.corrcoef(ee[j], k[j])[0, 1]:.4f}, rata-rata EE {ee[j].mean():.2f}, "
                  f"skrip {k[j].mean():.2f}, beda mutlak median {np.median(abs(ee[j] - k[j])):.3f}")
    if a.presip:
        out = agregasi(a.presip)
        out.to_csv(os.path.join(D, f"gfwed_komponen_{a.presip}.csv"), index=False)
        print(len(out), "baris")
