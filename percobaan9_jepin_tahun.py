"""
Percobaan 9: bootstrap gugus tahun secara eksak, buang-satu-tahun, dan kurva jumlah positif
diperpanjang — menjawab telaah panel kedua naskah JEPIN (Telaah/2026-10-14b-JEPIN).

  A. Enumerasi eksak bootstrap gugus tahun. Dengan 7 tahun uji hanya ada C(13, 7) = 1.716
     multiset tahun yang berbeda; masing-masing berbobot multinomial 7!/(prod k_i!) / 7^7.
     Peluang peringkat dihitung tepat, tanpa galat Monte Carlo. Juga: peluang GB juara menurut
     berapa kali 2022 terambil.
  B. Buang-satu-tahun: peringkat AUC-PR dan selisih GB - klimatologi bila satu tahun uji dibuang.
  C. Sama dengan A untuk ambang persentil 90 yang dihitung dari tahun latih tiap lipatan.
  D. Kurva jumlah positif, bootstrap iid berstrata dengan pengembalian, prevalensi asli, untuk
     23-276 positif (di atas 92 = proyeksi "data serupa yang lebih panjang").

Pakai:
    python percobaan9_jepin_tahun.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \\
        DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import itertools
from math import factorial
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(9)
LABEL = {"klimatologi": "Climatology", "gradient-boosting": "Gradient boosting",
         "persistence": "Persistence", "seasonal-naive": "Seasonal naive",
         "rasio-analog": "Ratio scaling", "regresi-ONI": "Logistic, ONI",
         "regresi-penuh": "Logistic, full"}

df, ambang = muat(panel_path, oni_path)
TAHUN = list(range(2019, 2026))


def kumpul(ambang_latih):
    ys, raws, ths = [], {}, []
    for th in TAHUN:
        d = df
        if ambang_latih:
            d = df.copy()
            amb = d.loc[d["tahun"] < th, "titik_panas"].quantile(0.90)
            d["y"] = (d["titik_panas"] > amb).astype(int)
        latih, uji = d[d["tahun"] < th], d[d["tahun"] == th]
        for nama, (sl, su) in skor_model(latih, uji).items():
            raws.setdefault(nama, []).append(su)
        ys.append(uji["y"].to_numpy()); ths.append(uji["tahun"].to_numpy())
    return np.concatenate(ys), {n: np.concatenate(v) for n, v in raws.items()}, np.concatenate(ths)


def peringkat(nilai, nama):
    urut = sorted(nama, key=lambda n: -nilai[n])
    return {n: urut.index(n) + 1 for n in nama}


def balik(p, acuan, nama):
    return sum((p[a] - p[b]) * (acuan[a] - acuan[b]) < 0 for a, b in itertools.combinations(nama, 2))


def eksak(y, raw, thn, judul):
    NAMA = list(raw)
    idx_t = {t: np.where(thn == t)[0] for t in TAHUN}
    acuan = peringkat({n: average_precision_score(y, raw[n]) for n in NAMA}, NAMA)
    URUT = sorted(NAMA, key=lambda n: acuan[n])
    P = np.zeros((7, 7)); sama = 0.0; tot = 0.0; brev = 0.0
    atas = {(a, b): 0.0 for a, b in zip(URUT[:-1], URUT[1:])}
    gb1_by22 = {}
    for kombi in itertools.combinations_with_replacement(TAHUN, 7):
        k = [kombi.count(t) for t in TAHUN]
        w = factorial(7) / np.prod([factorial(x) for x in k]) / 7 ** 7
        idx = np.concatenate([np.tile(idx_t[t], c) for t, c in zip(TAHUN, k) if c])
        v = {n: average_precision_score(y[idx], raw[n][idx]) for n in NAMA}
        r = peringkat(v, NAMA)
        for i, n in enumerate(URUT):
            P[i, r[n] - 1] += w
        sama += w * all(r[n] == acuan[n] for n in NAMA)
        brev += w * balik(r, acuan, NAMA)
        for (a, b) in atas:
            atas[(a, b)] += w * (v[a] > v[b])
        c22 = k[TAHUN.index(2022)]
        s = gb1_by22.setdefault(c22, [0.0, 0.0])
        s[0] += w * (r["gradient-boosting"] == 1); s[1] += w
        tot += w
    print("\n" + "=" * 90)
    print(f"{judul}: enumerasi eksak 1.716 multiset tahun (jumlah bobot {tot:.6f})")
    print("=" * 90)
    print(f"  positif {y.sum()} ({100 * y.mean():.1f}%)")
    print(f"  {'':<18}" + "".join(f"{k:>7}" for k in range(1, 8)))
    for i, n in enumerate(URUT):
        print(f"  {LABEL[n]:<18}" + "".join(f"{100 * P[i, k]:>7.1f}" for k in range(7)))
    print(f"  peringkat utuh sama: {100 * sama:.1f}%   rerata pembalikan: {brev:.2f}")
    for (a, b), p in atas.items():
        print(f"  P({LABEL[a]} > {LABEL[b]}) = {100 * p:.1f}%")
    print("  P(GB juara | 2022 terambil k kali):")
    for c in sorted(gb1_by22):
        s = gb1_by22[c]
        print(f"    k={c}: {100 * s[0] / s[1]:5.1f}%  (bobot {100 * s[1]:.1f}%)")
    return NAMA, acuan


# ---------------------------------------------------------------- A dan B

y, raw, thn = kumpul(False)
NAMA, acuan = eksak(y, raw, thn, "A. AMBANG PENUH")

print("\n" + "=" * 90)
print("B. BUANG SATU TAHUN UJI")
print("=" * 90)
for t in TAHUN:
    m = thn != t
    v = {n: average_precision_score(y[m], raw[n][m]) for n in NAMA}
    r = peringkat(v, NAMA)
    print(f"  tanpa {t}: positif {y[m].sum():>3}; GB {v['gradient-boosting']:.3f}, klim "
          f"{v['klimatologi']:.3f}, selisih {v['gradient-boosting'] - v['klimatologi']:+.3f}; "
          f"juara {LABEL[min(NAMA, key=lambda n: r[n])]}; pembalikan {balik(r, acuan, NAMA)}")

# ---------------------------------------------------------------- C

yL, rawL, thnL = kumpul(True)
eksak(yL, rawL, thnL, "C. AMBANG DARI TAHUN LATIH")

# ---------------------------------------------------------------- D

print("\n" + "=" * 90)
print("D. KURVA JUMLAH POSITIF (bootstrap iid berstrata dengan pengembalian, prevalensi asli, 300)")
print("=" * 90)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
URUT = sorted(NAMA, key=lambda n: acuan[n])
for p in [23, 46, 92, 184, 276]:
    nn = int(round(p * len(neg) / len(pos)))
    b, juara, gbk = [], [], []
    for _ in range(300):
        idx = np.concatenate([rng.choice(pos, size=p, replace=True), rng.choice(neg, size=nn, replace=True)])
        v = {n: average_precision_score(y[idx], raw[n][idx]) for n in NAMA}
        r = peringkat(v, NAMA)
        b.append(balik(r, acuan, NAMA)); juara.append(r[URUT[0]] == 1)
    b = np.array(b)
    print(f"  {p:>3} positif{' (proyeksi)' if p > 92 else '':<11}: pembalikan rerata {b.mean():.2f} "
          f"[{np.percentile(b, 2.5):.0f}, {np.percentile(b, 97.5):.0f}]; juara tetap {100 * np.mean(juara):.0f}%; "
          f"peringkat utuh sama {100 * np.mean(b == 0):.0f}%")
print("\nselesai.")
