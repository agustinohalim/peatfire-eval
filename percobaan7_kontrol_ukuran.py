"""
Percobaan 7: apakah ketidakstabilan peringkat berasal dari PENYEIMBANGAN, atau dari ukuran data uji?

Telaah panel 14 Oktober 2026 (Telaah/2026-10-14-JEPIN, temuan kritis editor, metode, dan devil's
advocate): pembalikan peringkat di sel A dihitung terhadap peringkat sel D seolah-olah sel D tanpa
galat, padahal sel D sendiri hanya bertumpu pada 92 positif. Satu undian seimbang hanya memuat
184 kabupaten-bulan (92 positif + 92 negatif). Sebagian ragam itu mungkin sekadar ragam sampel kecil.

Kontrol, masing-masing 300 ulangan, dibandingkan dengan peringkat tetap sel C (ROC-AUC) dan
sel D (AUC-PR) pada data uji utuh:

  seimbang          92 positif + 92 negatif tanpa pengembalian (= sel A/B, rancangan naskah)
  bootstrap-asli    1.176 kabupaten-bulan dengan pengembalian, prevalensi asli
  subsampel-184     184 kabupaten-bulan tanpa pengembalian, berstrata ke prevalensi asli
                    (14 positif + 170 negatif): ukuran sama dengan undian seimbang
  bootstrap-kab     14 kabupaten diambil ulang dengan pengembalian (gugus)
  bootstrap-tahun   7 tahun uji diambil ulang dengan pengembalian (gugus)

Pembalikan = jumlah pasangan model (dari 21) yang urutannya berbeda dari peringkat acuan.

Pakai:
    python percobaan7_kontrol_ukuran.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \\
        DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
ULANG = 300
rng = np.random.default_rng(7)

df, ambang = muat(panel_path, oni_path)
kum = {}
kab, thn = [], []
for th in range(2019, 2026):
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    kab.append(uji["kabupaten"].to_numpy()); thn.append(uji["tahun"].to_numpy())
    for nama, (sl, su) in skor_model(latih, uji).items():
        kum.setdefault(nama, {"y": [], "raw": []})
        kum[nama]["y"].append(uji["y"].to_numpy())
        kum[nama]["raw"].append(su)

NAMA = list(kum)
y = np.concatenate(kum[NAMA[0]]["y"])
raw = {n: np.concatenate(kum[n]["raw"]) for n in NAMA}
kab, thn = np.concatenate(kab), np.concatenate(thn)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
PASANGAN = list(itertools.combinations(NAMA, 2))


def peringkat(nilai):
    urut = sorted(NAMA, key=lambda n: -nilai[n])
    return {n: urut.index(n) + 1 for n in NAMA}


def balik(p, acuan):
    return sum((p[a] - p[b]) * (acuan[a] - acuan[b]) < 0 for a, b in PASANGAN)


acuanC = peringkat({n: roc_auc_score(y, raw[n]) for n in NAMA})
acuanD = peringkat({n: average_precision_score(y, raw[n]) for n in NAMA})

n_pos_sub = int(round(len(pos) * 184 / len(y)))
n_neg_sub = 184 - n_pos_sub
kab_u, thn_u = np.unique(kab), np.unique(thn)
idx_kab = {k: np.where(kab == k)[0] for k in kab_u}
idx_thn = {t: np.where(thn == t)[0] for t in thn_u}


def ambil(skema):
    if skema == "seimbang":
        return np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
    if skema == "bootstrap-asli":
        return rng.choice(len(y), size=len(y), replace=True)
    if skema == "subsampel-184":
        return np.concatenate([rng.choice(pos, size=n_pos_sub, replace=False),
                               rng.choice(neg, size=n_neg_sub, replace=False)])
    if skema == "bootstrap-kab":
        return np.concatenate([idx_kab[k] for k in rng.choice(kab_u, size=len(kab_u))])
    if skema == "bootstrap-tahun":
        return np.concatenate([idx_thn[t] for t in rng.choice(thn_u, size=len(thn_u))])
    raise ValueError(skema)


print(f"Panel: {len(df)} baris, ambang > {ambang:.0f}, uji {len(y)} kabupaten-bulan, "
      f"{len(pos)} positif ({y.mean() * 100:.1f}%), {len(NAMA)} model, {ULANG} ulangan")
print(f"subsampel berukuran sama: {n_pos_sub} positif + {n_neg_sub} negatif\n")

hasil = {}
for skema in ["seimbang", "bootstrap-asli", "subsampel-184", "bootstrap-kab", "bootstrap-tahun"]:
    bC, bD = [], []
    for _ in range(ULANG):
        while True:
            idx = ambil(skema)
            if 0 < y[idx].sum() < len(idx):
                break
        yy = y[idx]
        bC.append(balik(peringkat({n: roc_auc_score(yy, raw[n][idx]) for n in NAMA}), acuanC))
        bD.append(balik(peringkat({n: average_precision_score(yy, raw[n][idx]) for n in NAMA}), acuanD))
    hasil[skema] = (np.array(bC), np.array(bD))

print("=" * 96)
print("PEMBALIKAN PASANGAN (dari 21) TERHADAP PERINGKAT DATA UJI UTUH")
print("=" * 96)
print(f"{'skema':<18}{'ROC-AUC vs C: median [2.5,97.5]  nol':>40}{'AUC-PR vs D: median [2.5,97.5]  nol':>38}")
print("-" * 96)
for skema, (bC, bD) in hasil.items():
    def ring(b):
        return (f"{np.median(b):>4.0f} [{np.percentile(b, 2.5):.0f}, {np.percentile(b, 97.5):.0f}]"
                f"  rerata {b.mean():.2f}  nol {np.sum(b == 0):>3}/{ULANG}")
    print(f"{skema:<18}{ring(bC):>40}{ring(bD):>38}")
print("\nselesai.")
