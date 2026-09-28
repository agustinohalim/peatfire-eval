"""
Artikel Kalimantan — uji kepekaan: ambang sasaran dari tahun latih saja.

kalimantan_prediksi.py menghitung kedua ambang (gabungan p90 seluruh Kalimantan, dan p90 tiap
kabupaten) sekali dari 2012-2025, termasuk tahun uji. Di sini, untuk setiap lipatan, ambang
dihitung ulang dari tahun latih lipatan itu saja (rolling: 2012 .. th-1; ekstrem: semua tahun
selain th) lalu diterapkan ke baris latih dan uji. Model, fitur, dan benih sama persis
(bagian definisi kalimantan_prediksi.py dipakai lewat exec).

Pakai:
    python kalimantan_kepekaan_ambang.py DL_FIRE_NASIONAL/panel_kalimantan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt
    python kalimantan_selang.py DL_FIRE_NASIONAL/prediksi_kalimantan_kepekaan.csv

Keluaran: DL_FIRE_NASIONAL/prediksi_kalimantan_kepekaan.csv (tidak masuk repo, ± 15 menit)
"""

import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "kalimantan_prediksi.py"), encoding="utf-8").read().split("baris = []")[0])
OUT = os.path.join(os.path.dirname(jalur_kal), "prediksi_kalimantan_kepekaan.csv")


def label_ulang(d, tahun_ambang, sasaran):
    ref = mentah[mentah["tahun"].isin(tahun_ambang)]
    d = d.copy()
    if sasaran == "gabungan":
        d["y"] = (d["titik_panas"] > ref["titik_panas"].quantile(0.90)).astype(int)
    else:
        q = ref.groupby("kabupaten")["titik_panas"].quantile(0.90)
        d["y"] = (d["titik_panas"] > d["kabupaten"].map(q)).astype(int)
    return d


semua = sorted(mentah["tahun"].unique())
baris = []
for nama_s in ("gabungan", "kabupaten"):
    for th in range(2019, 2026):
        d = label_ulang(df, [t for t in semua if t < th], nama_s)
        lalu, uji = d[d["tahun"] < th], d[d["tahun"] == th]
        blok = uji[["provinsi", "kabupaten", "tahun", "bulan", "y"]].copy()
        blok["sasaran"], blok["skema"] = nama_s, "rolling"
        for n, (_, su) in skor_model(lalu, uji).items():
            blok[n] = su
        blok["gb_prov"] = gb(X(lalu, True), lalu["y"], X(uji, True))
        blok["gb_lopo"] = np.nan
        blok["gb_lokal"] = np.nan
        for p in PROV:
            mu = (uji["provinsi"] == p).to_numpy()
            lain, sendiri = lalu[lalu["provinsi"] != p], lalu[lalu["provinsi"] == p]
            blok.loc[mu, "gb_lopo"] = gb(X(lain), lain["y"], X(uji[mu]))
            if sendiri["y"].nunique() == 2:
                blok.loc[mu, "gb_lokal"] = gb(X(sendiri), sendiri["y"], X(uji[mu]))
        baris.append(blok)
        print(f"  {nama_s} {th} selesai; prevalensi uji {uji['y'].mean() * 100:.2f}%", flush=True)
    for th in (2015, 2014):
        d = label_ulang(df, [t for t in semua if t != th], nama_s)
        sisa, uji = d[d["tahun"] != th], d[d["tahun"] == th]
        blok = uji[["provinsi", "kabupaten", "tahun", "bulan", "y"]].copy()
        blok["sasaran"], blok["skema"] = nama_s, f"ekstrem_{th}"
        for n, (_, su) in skor_model(sisa, uji).items():
            blok[n] = su
        baris.append(blok)
        print(f"  {nama_s} ekstrem {th} selesai", flush=True)

pd.concat(baris, ignore_index=True).to_csv(OUT, index=False)
print(f"tersimpan: {OUT}")
