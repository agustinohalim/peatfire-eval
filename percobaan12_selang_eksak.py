"""
Percobaan 12: selang 95 % eksak (enumerasi 1.716 multiset tahun) untuk AUC-PR dan ROC-AUC tiap
model, menggantikan selang Monte Carlo 1.000 undian di Tabel II naskah JEPIN (telaah panel ketiga,
metode M6).

Pakai:
    python percobaan12_selang_eksak.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \
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
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])
df, ambang = muat(sys.argv[1], sys.argv[2])
TAHUN = list(range(2019, 2026))
ys, raws, ths = [], {}, []
for th in TAHUN:
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    for nama, (sl, su) in skor_model(latih, uji).items():
        raws.setdefault(nama, []).append(su)
    ys.append(uji["y"].to_numpy()); ths.append(uji["tahun"].to_numpy())
y = np.concatenate(ys); raw = {n: np.concatenate(v) for n, v in raws.items()}; thn = np.concatenate(ths)
idx_t = {t: np.where(thn == t)[0] for t in TAHUN}
W, PR, RC = [], {n: [] for n in raw}, {n: [] for n in raw}
for kombi in itertools.combinations_with_replacement(TAHUN, 7):
    k = [kombi.count(t) for t in TAHUN]
    W.append(factorial(7) / np.prod([factorial(x) for x in k]) / 7 ** 7)
    idx = np.concatenate([np.tile(idx_t[t], c) for t, c in zip(TAHUN, k) if c])
    for n in raw:
        PR[n].append(average_precision_score(y[idx], raw[n][idx]))
        RC[n].append(roc_auc_score(y[idx], raw[n][idx]))
W = np.array(W)


def q(x, p):
    x = np.asarray(x); o = np.argsort(x); c = np.cumsum(W[o]) / W.sum()
    return x[o][np.searchsorted(c, p)]


for n in sorted(raw, key=lambda n: -average_precision_score(y, raw[n])):
    print(f"  {n:<18} AUC-PR {average_precision_score(y, raw[n]):.3f} [{q(PR[n], .025):.3f}, {q(PR[n], .975):.3f}]"
          f"  ROC-AUC {roc_auc_score(y, raw[n]):.3f} [{q(RC[n], .025):.3f}, {q(RC[n], .975):.3f}]")
print("selesai.")
