"""
Menggabungkan Fire Weather Index bulanan (GFWED, Skrip_FWI_Earth_Engine.js) ke panel fitur
nasional PeatFireBench yang sudah memuat gambut.

Masukan : DL_FIRE_NASIONAL/fwi_bulanan_kabupaten_2012_2025_v2.csv  (GID_2, bulan, fwi, fwi_maks,
                                                                 fwi_asli, hari)
          DL_FIRE_NASIONAL/panel_nasional_fitur_gambut.csv       (gabung_gambut_nasional.py)
          DL_FIRE_SV-C2_792597/gadm41_IDN_2.json                 (pusat kabupaten, untuk pengisian)
Keluaran: DL_FIRE_NASIONAL/panel_nasional_fitur_penuh.csv

Pengisian sel laut. GFWED hanya berisi sel daratan MERRA-2 (±55 km). Kabupaten pulau dan pesisir
sempit yang tidak menutupi sel daratan diisi di Earth Engine dari rata-rata sel daratan dalam
dua sel (fwi_asli null, fwi terisi). Yang masih kosong sesudah itu (dua kabupaten pada contoh
September 2015) diisi di sini dari kabupaten terdekat menurut jarak pusat. Semuanya ditandai
fwi_imputasi = 1 (1 = diisi di Earth Engine, 2 = diisi dari tetangga).

Fitur (kelompok "cuaca" di kelompok_fitur.py), semuanya jeda; FWI bulan berjalan adalah cuaca
serentak dengan kebakaran dan tidak dipakai sebagai fitur:
  fwi_lag1..3      : rata-rata FWI bulan t-1..t-3
  fwi_maks_lag1    : FWI harian tertinggi bulan t-1
  fwi_anom_lag1    : fwi_lag1 dikurangi rata-rata kabupaten-bulan-kalender 2012-2018 (garis dasar
                     berhenti sebelum tahun uji kronologis, sama dengan hujan)
Kolom rujukan, BUKAN fitur: fwi_kini (FWI bulan t) untuk baris rujukan "cuaca sempurna".

Pakai:  python gabung_fwi_nasional.py
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
GADM = os.path.join(BASE, "DL_FIRE_SV-C2_792597", "gadm41_IDN_2.json")
# Hasil pencarian tetangga terdekat pada 2 Oktober 2026, dipakai bila poligon GADM tidak ada
# (repo publik tidak menyertakannya): {kabupaten kosong: kabupaten pengisi}.
TETANGGA_TERCATAT = {"IDN.19.7_1": "IDN.19.11_1", "IDN.29.6_1": "IDN.29.7_1"}


def pusat(g):
    """Pusat kasar: rata-rata titik cincin luar poligon terbesar (cukup untuk tetangga terdekat)."""
    pol = g["coordinates"] if g["type"] == "Polygon" else max(g["coordinates"], key=lambda p: len(p[0]))
    a = np.asarray(pol[0])
    return a[:, 0].mean(), a[:, 1].mean()


c = pd.read_csv(os.path.join(D, "fwi_bulanan_kabupaten_2012_2025_v2.csv")).rename(columns={"GID_2": "gid"})
c = c[["gid", "bulan", "fwi", "fwi_maks", "fwi_asli", "hari"]]
assert len(c) == 502 * 168 and not c.duplicated(["gid", "bulan"]).any(), len(c)
assert c["hari"].between(28, 31).all(), c["hari"].describe()
c["fwi_imputasi"] = c["fwi_asli"].isna().astype(int)
kosong = sorted(set(c.loc[c["fwi"].isna(), "gid"]))
if kosong:
    if os.path.exists(GADM):
        gj = json.load(open(GADM, encoding="utf-8"))
        xy = {f["properties"]["GID_2"]: pusat(f["geometry"]) for f in gj["features"]}
        terisi = [g for g in xy if g not in kosong]
        dekat = {g: min(terisi, key=lambda h: (xy[h][0] - xy[g][0]) ** 2 + (xy[h][1] - xy[g][1]) ** 2)
                 for g in kosong}
        assert all(dekat[g] == TETANGGA_TERCATAT.get(g, dekat[g]) for g in kosong), dekat
    else:
        assert set(kosong) == set(TETANGGA_TERCATAT), kosong
        dekat = TETANGGA_TERCATAT
    for g in kosong:
        t = dekat[g]
        isi = c[c["gid"] == t].set_index("bulan")
        b = c["gid"] == g
        for k in ("fwi", "fwi_maks"):
            c.loc[b, k] = c.loc[b, "bulan"].map(isi[k]).to_numpy()
        c.loc[b, "fwi_imputasi"] = 2
        print(f"  {g}: kosong di {int(b.sum())} bulan, diisi dari {t}")
assert c["fwi"].notna().all() and c["fwi_maks"].notna().all()
print(f"satuan terisi di Earth Engine: {c.loc[c.fwi_imputasi == 1, 'gid'].nunique()}; "
      f"dari tetangga: {len(kosong)}")

c["tahun"] = c["bulan"].str[:4].astype(int)
c["bulan_ke"] = c["bulan"].str[5:7].astype(int)
c = c.sort_values(["gid", "bulan"])
normal = c[c["tahun"] <= 2018].groupby(["gid", "bulan_ke"])["fwi"].mean().rename("normal")
c = c.join(normal, on=["gid", "bulan_ke"])
g = c.groupby("gid")
for l in (1, 2, 3):
    c[f"fwi_lag{l}"] = g["fwi"].shift(l)
c["fwi_maks_lag1"] = g["fwi_maks"].shift(1)
c["fwi_anom_lag1"] = c["fwi_lag1"] - g["normal"].shift(1)
c["fwi_kini"] = c["fwi"]

f = pd.read_csv(os.path.join(D, "panel_nasional_fitur_gambut.csv"))
kol = ["gid", "bulan", "fwi_imputasi", "fwi_lag1", "fwi_lag2", "fwi_lag3", "fwi_maks_lag1",
       "fwi_anom_lag1", "fwi_kini"]
out = f.merge(c[kol], on=["gid", "bulan"], how="left", validate="one_to_one")
assert len(out) == len(f) and out[kol[2:]].notna().all().all(), out[kol[2:]].isna().sum()
out.to_csv(os.path.join(D, "panel_nasional_fitur_penuh.csv"), index=False)
print(f"{len(out)} baris, {out['gid'].nunique()} satuan; kolom baru: {', '.join(kol[2:])}")
for k in ("fwi_lag1", "fwi_anom_lag1", "fwi_kini"):
    print(f"  Spearman {k} vs titik_panas: {out[[k, 'titik_panas']].corr('spearman').iloc[0, 1]:.3f}")
s = out[out["bulan"] == "2015-09"].groupby("provinsi")["fwi_kini"].mean().sort_values(ascending=False)
print("FWI September 2015 per provinsi (lima tertinggi):", s.head(5).round(1).to_dict())
