"""
Artikel 2 (PeatFireBench) — uji kepekaan: ambang sasaran dari tahun latih saja.

Ambang baku (bersihkan_panel_nasional.py) dihitung sekali dari 2012-2025, termasuk tahun uji.
Itu kebocoran label yang kecil tetapi nyata (REFORMS). Di sini, untuk setiap lipatan, ambang
dihitung ulang hanya dari tahun latih lipatan itu dan diterapkan ke baris latih dan uji:

  S1 kronologis   : uji th (2019-2025), ambang dari panel 2012 .. th-1
  S2 tahun ekstrem: uji th (2015, 2014, 2019), ambang dari semua tahun selain th

  y_kabupaten = titik_panas > maks(persentil 90 kabupaten dari tahun ambang, 10)
  y_gabungan  = titik_panas > persentil 90 seluruh satuan dari tahun ambang

Himpunan kabupaten yang dinilai (ada_positif) dibiarkan sama dengan analisis baku, supaya yang
berubah hanya ambangnya. Model, fitur (PATOKAN_FITUR=lengkap), dan parameter sama dengan
patokan_prediksi.py (dipakai langsung lewat exec bagian definisinya). S3 tidak diulang.

Pakai:    python patokan_kepekaan_ambang.py      (± 30 menit; jalankan di latar belakang)
Lalu:     PATOKAN_BERKAS=patokan_prediksi_kepekaan.csv python patokan_tabel_naskah.py
Keluaran: DL_FIRE_NASIONAL/patokan_prediksi_kepekaan.csv (tidak masuk repo)
"""

import os

import numpy as np
import pandas as pd

os.environ["PATOKAN_FITUR"] = "lengkap"
BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "patokan_prediksi.py"), encoding="utf-8").read().split("baris = []")[0])

OUT = os.path.join(D, "patokan_prediksi_kepekaan.csv")
BATAS_BAWAH = 10
panel = pd.read_csv(os.path.join(D, "panel_nasional_bersih.csv"), usecols=["gid", "tahun", "titik_panas"])


def label_ulang(d, tahun_ambang):
    ref = panel[panel["tahun"].isin(tahun_ambang)]
    q_kab = np.maximum(ref.groupby("gid")["titik_panas"].quantile(0.90), BATAS_BAWAH)
    gab = ref["titik_panas"].quantile(0.90)
    d = d.copy()
    d["y_kabupaten"] = (d["titik_panas"] > d["gid"].map(q_kab)).astype(int)
    d["y_gabungan"] = (d["titik_panas"] > gab).astype(int)
    return d, gab


baris = []


def simpan(uji, skema, ylab, tag, skor):
    b = uji[["gid", "provinsi", "pulau", "tahun", "bulan", "ada_positif"]].copy()
    b["y"] = uji[ylab].to_numpy()
    b["sasaran"], b["skema"], b["tag"] = ylab, skema, tag
    for n, v in skor.items():
        b[n] = v
    baris.append(b)


semua = sorted(panel["tahun"].unique())
for skema, tahun_uji in (("S1", range(2019, 2026)), ("S2", (2015, 2014, 2019))):
    for th in tahun_uji:
        ambang_thn = [t for t in semua if (t < th if skema == "S1" else t != th)]
        d, gab = label_ulang(df, ambang_thn)
        latih = d[d["tahun"] < th] if skema == "S1" else d[d["tahun"] != th]
        uji = d[d["tahun"] == th]
        for ylab in ("y_kabupaten", "y_gabungan"):
            simpan(uji, skema, ylab, str(th), {**lokal(latih, uji), **dipelajari(latih, uji, ylab)})
        print(f"{skema} {th}: ambang dari {ambang_thn[0]}-{ambang_thn[-1]} (tanpa {th})" if skema == "S2" else
              f"{skema} {th}: ambang dari {ambang_thn[0]}-{ambang_thn[-1]}", f"gabungan > {gab:.0f}",
              f"prevalensi uji kab {uji['y_kabupaten'].mean() * 100:.2f}% gab {uji['y_gabungan'].mean() * 100:.2f}%",
              flush=True)

pd.concat(baris, ignore_index=True).to_csv(OUT, index=False)
print(f"tersimpan: {OUT}")
