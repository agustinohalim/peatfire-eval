"""
Artikel 2 (PeatFireBench) — uji pembalikan tahun ekstrem pada himpunan tahun yang DITETAPKAN
(panel telaah kedua, Telaah/PeatFireBench_2026-10-02: kontras sebelumnya memakai 2015, 2019, 2023
dan membuang 2014).

Himpunan tetap: tahun ekstrem = S2 2015, 2014, 2019 (Bagian 4.1); tahun tenang = S1 2020, 2021,
2022, 2024, 2025. 2023 dilaporkan terpisah sebagai kepekaan. 411 kabupaten yang sama untuk kedua
sasaran. Model: gradient boosting, random forest, MLP, dan dua baseline jeda (persistence,
seasonal naive) — apakah "model yang dipelajari membantu" sebenarnya "jeda membantu"?

Untuk tiap model dan sasaran: kontras ekstrem - tenang dari keunggulan atas klimatologi, mentah
dan sebagai bagian ruang sisa (AP_m - AP_k) / (1 - AP_k); selang 95 % bootstrap kabupaten
berpasangan (300 ulangan); dan rentang jackknife tahun ekstrem (kontras bila satu tahun ekstrem
dibuang) sebagai ukuran ketergantungan pada tahun tertentu.

Pakai:    python patokan_uji_tahun.py
Keluaran: layar (simpan ke DL_FIRE_NASIONAL/oni_akhir/patokan_uji_tahun.log)
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
P = pd.read_csv(os.path.join(D, "patokan_prediksi_lengkap.csv"), dtype={"tag": str})
P = P[P.ada_positif == 1]
EKSTREM = [("S2", "2015"), ("S2", "2014"), ("S2", "2019")]
TENANG = [("S1", t) for t in ("2020", "2021", "2022", "2024", "2025")]
MODEL = ["xgb", "rf", "mlp", "persistence", "seasonal_naive"]
SAS = ("y_kabupaten", "y_gabungan")
rng = np.random.default_rng(2026)

blok = {(s, sk, t): P[(P.sasaran == s) & (P.skema == sk) & (P.tag == t)]
        for s in SAS for sk, t in EKSTREM + TENANG + [("S1", "2023")]}
kab = P.gid.unique()
idx = {k: {g: np.where(b.gid.to_numpy() == g)[0] for g in kab} for k, b in blok.items()}


def gain(key, m, pilih, ruang):
    b = blok[key]
    ii = np.concatenate([idx[key][g] for g in pilih]) if pilih is not None else np.arange(len(b))
    y = b.y.to_numpy()[ii]
    if y.sum() == 0:
        return np.nan
    a, k = ap(y, b[m].to_numpy()[ii]), ap(y, b.klimatologi.to_numpy()[ii])
    return (a - k) / (1 - k) if ruang else a - k


def kontras(s, m, pilih=None, ruang=True, ekstrem=EKSTREM):
    e = np.nanmean([gain((s,) + x, m, pilih, ruang) for x in ekstrem])
    q = np.nanmean([gain((s,) + x, m, pilih, ruang) for x in TENANG])
    return e - q


for ruang in (False, True):
    print(f"\n=== {'bagian ruang sisa' if ruang else 'selisih AUC-PR mentah'}; ekstrem = S2 2015, 2014, 2019")
    print(f"  {'model':16s}{'per kabupaten':>30s}{'gabungan':>30s}{'selisih antar-sasaran':>32s}  jackknife (buang 1 thn ekstrem)")
    for m in MODEL:
        titik = {s: kontras(s, m, ruang=ruang) for s in SAS}
        bs = np.array([[kontras(s, m, rng.choice(kab, len(kab)), ruang) for s in SAS] for _ in range(300)])
        jk = [kontras("y_kabupaten", m, ruang=ruang, ekstrem=[x for x in EKSTREM if x != y]) -
              kontras("y_gabungan", m, ruang=ruang, ekstrem=[x for x in EKSTREM if x != y]) for y in EKSTREM]
        teks = [f"{titik[s]:+.3f} [{np.percentile(bs[:, i], 2.5):+.3f}, {np.percentile(bs[:, i], 97.5):+.3f}]"
                for i, s in enumerate(SAS)]
        d = bs[:, 0] - bs[:, 1]
        print(f"  {m:16s}{teks[0]:>30s}{teks[1]:>30s}"
              f"{titik['y_kabupaten'] - titik['y_gabungan']:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
              f"   {min(jk):+.3f} .. {max(jk):+.3f}")

print("\n=== per tahun, gradient boosting dan persistence, bagian ruang sisa")
for s in SAS:
    for sk, t in EKSTREM + [("S1", "2023")] + TENANG:
        print(f"  {s:12s} {sk} {t}  GB {gain((s, sk, t), 'xgb', None, True):+.3f}  "
              f"persistence {gain((s, sk, t), 'persistence', None, True):+.3f}  "
              f"prev {blok[(s, sk, t)].y.mean() * 100:5.1f}%")

print("\n=== kepekaan: ekstrem = S2 2015, 2019 dan S1 2023 (himpunan pada versi naskah sebelumnya)")
for m in ("xgb", "persistence"):
    v = [kontras(s, m, ekstrem=[("S2", "2015"), ("S2", "2019"), ("S1", "2023")]) for s in SAS]
    print(f"  {m:12s} per kabupaten {v[0]:+.3f}  gabungan {v[1]:+.3f}  selisih {v[0] - v[1]:+.3f}")
