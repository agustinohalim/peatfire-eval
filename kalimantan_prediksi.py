"""
Menghitung dan menyimpan seluruh prediksi artikel Kalimantan sekali saja, supaya selang
bootstrap dan gambar membaca berkas yang sama dan tidak melatih ulang model.

Isi keluaran (satu baris per kabupaten-bulan uji, per sasaran, per skema):
  sasaran  : "gabungan" (p90 seluruh Kalimantan) atau "kabupaten" (p90 tiap kabupaten)
  skema    : "rolling" (uji 2019-2025, latih tahun sebelumnya) atau "ekstrem_2015"/"ekstrem_2014"
             (tahun itu disisihkan penuh dari pelatihan, dilatih pada semua tahun lain)
  kolom model:
    tujuh model Artikel 1, dilatih pada seluruh Kalimantan (ID)
    gb_prov  : gradient boosting ID + satu-panas provinsi
    gb_lopo  : gradient boosting dilatih tanpa provinsi uji
    gb_lokal : gradient boosting dilatih pada provinsi uji saja (NaN bila latih satu kelas)
  Skema ekstrem hanya memuat tujuh model ID.

Angka yang dihasilkan sama dengan percobaan_kalimantan*.py karena model, fitur, dan benih acak
sama (skor_model dari percobaan.py; gradient boosting dengan parameter yang sama).

Pakai:
    python kalimantan_prediksi.py DL_FIRE_NASIONAL/panel_kalimantan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt

Keluaran: DL_FIRE_NASIONAL/prediksi_kalimantan.csv (tidak masuk repo, ± 15 menit)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py"), encoding="utf-8").read().split("if __name__")[0])

jalur_kal, oni_path = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.path.dirname(jalur_kal), "prediksi_kalimantan.csv")
df, ambang = muat(jalur_kal, oni_path)
PROV = sorted(df["provinsi"].unique())
mentah = pd.read_csv(jalur_kal)
q_kab = mentah[mentah["tahun"] <= 2025].groupby("kabupaten")["titik_panas"].quantile(0.90)
d_kab = df.copy()
d_kab["y"] = (d_kab["titik_panas"] > d_kab["kabupaten"].map(q_kab)).astype(int)


def gb(Xl, yl, Xu):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                      random_state=42, verbosity=0)
    m.fit(Xl, yl)
    return m.predict_proba(Xu)[:, 1]


def X(d, prov=False):
    x = d[FITUR].fillna(0.0).copy()
    if prov:
        for p in PROV:
            x["prov_" + p] = (d["provinsi"] == p).astype(float)
    return x.to_numpy()


baris = []
for nama_s, d in (("gabungan", df), ("kabupaten", d_kab)):
    for th in range(2019, 2026):
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
        print(f"  {nama_s} {th} selesai", flush=True)
    for th in (2015, 2014):
        sisa, uji = d[d["tahun"] != th], d[d["tahun"] == th]
        blok = uji[["provinsi", "kabupaten", "tahun", "bulan", "y"]].copy()
        blok["sasaran"], blok["skema"] = nama_s, f"ekstrem_{th}"
        for n, (_, su) in skor_model(sisa, uji).items():
            blok[n] = su
        baris.append(blok)
        print(f"  {nama_s} ekstrem {th} selesai", flush=True)

hasil = pd.concat(baris, ignore_index=True)
hasil.to_csv(OUT, index=False)
print(f"tersimpan: {OUT}  baris: {len(hasil)}")
