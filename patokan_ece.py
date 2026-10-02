"""
PeatFireBench — ECE dengan selang dan per tahun (panel telaah ketiga, methods:M9, devils-advocate:X8).

ECE gradient boosting (sepuluh bin lebar sama, seperti patokan_tabel_naskah.py) untuk tahun biasa
S1 2020-2025 dan tahun ekstrem S2 digabung, dengan selang 95 % bootstrap kabupaten (1.000 ulangan);
lalu per tahun: S1 2019 dan S2 2019 (tahun sama, disisihkan atau tidak), S1 2023, dan tiap S2.
Juga prevalensi latih, supaya pergeseran prior terlihat: bila ECE tahun ekstrem sekadar akibat
prevalensi, tahun S1 berprevalensi tinggi (2019, 2023) mestinya setinggi S2.

Pakai:    python patokan_ece.py   (keluaran ke layar; DL_FIRE_NASIONAL/penuh/patokan_ece.log)
"""
import os
import numpy as np
import pandas as pd
from kelompok_fitur import PREDIKSI

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
P = pd.read_csv(os.path.join(D, PREDIKSI), dtype={"tag": str})
rng = np.random.default_rng(2026)


def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())


def boot(d, n=1000):
    g = d.gid.to_numpy(); uk = np.unique(g); idx = {k: np.where(g == k)[0] for k in uk}
    y, p = d.y.to_numpy(), d.xgb.to_numpy(); out = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(uk, len(uk))])
        out.append(ece(y[j], p[j]))
    return np.percentile(out, [2.5, 97.5])


for sas in ("y_kabupaten", "y_gabungan"):
    d0 = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    print(f"\n=== {sas}  (gradient boosting)")
    for nama, d in (("S1 2020-2025", d0[(d0.skema == "S1") & d0.tag.isin([str(t) for t in range(2020, 2026)])]),
                    ("S2 2015+2014+2019", d0[d0.skema == "S2"])):
        lo, hi = boot(d)
        print(f"  {nama:20s} ECE {ece(d.y.to_numpy(), d.xgb.to_numpy()):.3f} [{lo:.3f}, {hi:.3f}]  "
              f"prevalensi {d.y.mean() * 100:.1f}%  rata-rata prediksi {d.xgb.mean() * 100:.1f}%")
    for sk, t in (("S1", "2019"), ("S2", "2019"), ("S1", "2023"), ("S2", "2015"), ("S2", "2014"),
                  ("S1", "2020"), ("S1", "2024")):
        d = d0[(d0.skema == sk) & (d0.tag == t)]
        lo, hi = boot(d, 300)
        print(f"  {sk} {t:15s} ECE {ece(d.y.to_numpy(), d.xgb.to_numpy()):.3f} [{lo:.3f}, {hi:.3f}]  "
              f"prevalensi {d.y.mean() * 100:.1f}%  rata-rata prediksi {d.xgb.mean() * 100:.1f}%")
