"""
Percobaan 10: proyeksi rekaman lebih panjang dengan mengambil ulang TAHUN uji, bukan bulan.

Telaah panel ketiga naskah JEPIN (Telaah/2026-10-14c-JEPIN, editor X1): proyeksi percobaan 9 bagian D
mengambil ulang kabupaten-bulan secara bebas, padahal ketidakpastian peringkat datang dari tahun
(terutama 2022). Di sini rekaman uji sepanjang m tahun dibentuk dengan menarik m tahun dari tujuh
tahun uji yang ada, dengan pengembalian, 2.000 kali, untuk m = 7, 14, 21, 28.

Pakai:
    python percobaan10_proyeksi_tahun.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \
        DL_FIRE_SV-C2_792597/oni.ascii.txt
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])
panel_path, oni_path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(10)
ULANG = 2000

df, ambang = muat(panel_path, oni_path)
TAHUN = list(range(2019, 2026))
ys, raws, ths = [], {}, []
for th in TAHUN:
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    for nama, (sl, su) in skor_model(latih, uji).items():
        raws.setdefault(nama, []).append(su)
    ys.append(uji["y"].to_numpy()); ths.append(uji["tahun"].to_numpy())
y = np.concatenate(ys); raw = {n: np.concatenate(v) for n, v in raws.items()}; thn = np.concatenate(ths)
NAMA = list(raw)
idx_t = {t: np.where(thn == t)[0] for t in TAHUN}


def peringkat(v):
    urut = sorted(NAMA, key=lambda n: -v[n])
    return {n: urut.index(n) + 1 for n in NAMA}


acuan = peringkat({n: average_precision_score(y, raw[n]) for n in NAMA})
juara = min(NAMA, key=lambda n: acuan[n])
print(f"positif {y.sum()} dalam 7 tahun uji; {ULANG} ulangan per panjang rekaman")
for m in (7, 14, 21, 28):
    b, j, sama, gbk, pos = [], [], [], [], []
    for _ in range(ULANG):
        pilih = rng.choice(TAHUN, size=m, replace=True)
        idx = np.concatenate([idx_t[t] for t in pilih])
        v = {n: average_precision_score(y[idx], raw[n][idx]) for n in NAMA}
        r = peringkat(v)
        bb = sum((r[a] - r[c]) * (acuan[a] - acuan[c]) < 0 for a, c in itertools.combinations(NAMA, 2))
        b.append(bb); j.append(r[juara] == 1); sama.append(bb == 0)
        gbk.append(v["gradient-boosting"] > v["klimatologi"]); pos.append(y[idx].sum())
    print(f"  {m:>2} tahun (rerata {np.mean(pos):.0f} positif): juara tetap {100 * np.mean(j):.1f}%; "
          f"GB > klim {100 * np.mean(gbk):.1f}%; peringkat utuh sama {100 * np.mean(sama):.1f}%; "
          f"rerata pembalikan {np.mean(b):.2f}")
print("\nselesai.")
