"""
PeatFireBench revisi fase B5 — Temuan 2 terhadap rujukan klimatologi terkuat (Hasil_Patokan_7).

S1 2020-2025, kedua sasaran. Untuk tiap model: AUC-PR dalam-tahun dan gabungan antartahun; keunggulan
gabungan atas rujukan [95 %, bootstrap kabupaten 1.000] dan keunggulan dalam-tahun per tahun (apakah
keunggulan gabungan melebihi semua tahun tunggal).

Pakai:    python patokan_revisi_gabung.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_gabung.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
SEED = 2026
BIASA = [str(t) for t in range(2020, 2026)]
RUJUKAN = {"y_kabupaten": "klim_frek1", "y_gabungan": "klim_frek"}
MODEL = ["rf", "xgb", "mlp", "log_klim_jeda", "persistence", "fwi_logistik", "logistik", "klimatologi", "klim_frek", "klim_frek1"]

P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
P = P[(P.skema == "S1") & P.tag.isin(BIASA)]


def boot(d, a, b, n=1000):
    uk = d.gid.unique(); rng = np.random.default_rng(SEED)
    idx = {k: np.where(d.gid.to_numpy() == k)[0] for k in uk}
    y, pa, pb = d.y.to_numpy(), d[a].to_numpy(), d[b].to_numpy()
    out = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(uk, len(uk))])
        out.append(ap(y[j], pa[j]) - ap(y[j], pb[j]))
    return np.percentile(out, [2.5, 97.5])


for sas, ref in RUJUKAN.items():
    d = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d = d[d.ada_positif == 1]
    print(f"\n=== {sas}, rujukan {ref}: {d.gid.nunique()} kabupaten, {len(d)} baris")
    print(f"  {'model':<14}{'dalam':>7}{'gabung':>8}   unggul gabungan atas rujukan [95 %]     unggul dalam-tahun per tahun 2020..2025 (maks)")
    for m in MODEL:
        dal = np.mean([ap(g.y, g[m]) for _, g in d.groupby("tag")])
        gab = ap(d.y, d[m])
        per = [ap(g.y, g[m]) - ap(g.y, g[ref]) for _, g in d.groupby("tag")]
        s = ""
        if m != ref:
            lo, hi = boot(d, m, ref)
            s = f"{gab - ap(d.y, d[ref]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"  {m:<14}{dal:>7.3f}{gab:>8.3f}   {s:<38}" + ("" if m == ref else " ".join(f"{x:+.3f}" for x in per) + f" ({max(per):+.3f})"))
print("\nselesai.")
