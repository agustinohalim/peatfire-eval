"""
PeatFireBench revisi fase C7 (Hasil_Patokan_8): kalibrasi ulang Platt pada prakiraan luar-lipatan tahun
latih sesudah 2015 saja, dibandingkan dengan Platt semua tahun dan peluang mentah. Gradient boosting,
per kabupaten.

Pakai:    python -u patokan_revisi_pasca2015.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_pasca2015.log
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B

P = B.muat_P(fwi=False)
P = P[(P.sasaran == "y_kabupaten") & (P.ada_positif == 1)]
O = pd.read_csv(f"{B.D}/patokan_revisi_oof.csv", dtype={"tag": str})
O = O[O.ada_positif == 1]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())


def platt(o):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(o.xgb.to_numpy())[:, None], o.y.to_numpy())
    return lambda p: m.predict_proba(logit(p)[:, None])[:, 1]


print("=" * 100 + "\nC7. Platt pasca-2015 lawan Platt semua tahun lawan mentah, per kabupaten, gradient boosting\n" + "=" * 100)
hasil = {}
for sk, tags in (("S1", B.BIASA), ("S2", B.EKSTREM)):
    for t in tags:
        u = P[(P.skema == sk) & (P.tag == t)]; o = O[(O.skema == sk) & (O.tag == t)]
        y, p = u.y.to_numpy(), u.xgb.to_numpy()
        o16 = o[o.tahun_dalam >= 2016]
        v = {"mentah": p, "platt_semua": platt(o)(p), "platt_pasca2015": platt(o16)(p)}
        hasil[(sk, t)] = {k: (ece(y, q), q.mean()) for k, q in v.items()}
        print(f"  {sk} {t}: prevalensi {y.mean() * 100:.2f}%  tahun dalam pasca-2015 {o16.tahun_dalam.nunique()}  "
              + "  ".join(f"{k}: ECE {e:.4f} rata-rata p {m * 100:.2f}%" for k, (e, m) in hasil[(sk, t)].items()), flush=True)
for sk, tags, nama in (("S1", B.BIASA, "biasa"), ("S2", B.EKSTREM, "ekstrem")):
    print(f"\n  rata-rata tahun {nama}: " + "  ".join(
        f"{k} ECE {np.mean([hasil[(sk, t)][k][0] for t in tags]):.4f}" for k in ("mentah", "platt_semua", "platt_pasca2015")))
    prev = {t: P[(P.skema == sk) & (P.tag == t)].y.mean() for t in tags}
    lebih = sum(abs(hasil[(sk, t)]["platt_pasca2015"][1] - prev[t]) < abs(hasil[(sk, t)]["mentah"][1] - prev[t]) for t in tags)
    print(f"  tahun dengan |rata-rata p - prevalensi| lebih kecil (pasca-2015 lawan mentah): {lebih} dari {len(tags)}")
print("\nselesai.")
