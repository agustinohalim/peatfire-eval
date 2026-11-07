"""
PeatFireBench revisi fase B6 — Tabel 2 naskah revisi (Hasil_Patokan_7).

S1 2020-2025, kedua sasaran, semua baseline: AUC-PR dalam-tahun, keunggulan dalam-tahun atas rujukan
klimatologi terkuat [95 %, bootstrap kabupaten 1.000, tarikan sama untuk semua tahun], AUC-PR gabungan.
FWI persentil kabupaten dan klimatologi + FWI dari patokan_fwi_adil.csv.

Pakai:    python patokan_revisi_tabel2.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_tabel2.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
SEED = 2026
BIASA = [str(t) for t in range(2020, 2026)]
RUJUKAN = {"y_kabupaten": "klim_frek1", "y_gabungan": "klim_frek"}
MODEL = ["klimatologi", "klim_frek", "klim_frek1", "persistence", "seasonal_naive", "rasio", "fwi", "fwi_persentil",
         "fwi_logistik", "klim_fwi", "logistik", "log_klim_jeda", "rf", "xgb", "mlp"]
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
P = P[(P.skema == "S1") & P.tag.isin(BIASA)]
F = pd.read_csv(os.path.join(D, "patokan_fwi_adil.csv"), dtype={"tag": str})
P = P.merge(F[kunci + ["fwi_persentil", "klim_fwi"]], on=kunci, how="left", validate="one_to_one")
assert P[["fwi_persentil", "klim_fwi"]].notna().all().all()


def boot(d, a, b, n=1000):
    uk = d.gid.unique(); rng = np.random.default_rng(SEED)
    per = {t: g for t, g in d.groupby("tag")}
    idx = {t: {k: np.where(g.gid.to_numpy() == k)[0] for k in uk} for t, g in per.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, len(uk)); v = []
        for t, g in per.items():
            j = np.concatenate([idx[t][k] for k in pilih]); y = g.y.to_numpy()[j]
            if y.sum():
                v.append(ap(y, g[a].to_numpy()[j]) - ap(y, g[b].to_numpy()[j]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


for sas, ref in RUJUKAN.items():
    d = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d = d[d.ada_positif == 1]
    print(f"\n=== {sas}, rujukan {ref}: {d.gid.nunique()} kabupaten, {len(d)} baris, {int(d.y.sum())} positif, prevalensi {d.y.mean():.2%}")
    ar = {t: ap(g.y, g[ref]) for t, g in d.groupby("tag")}
    for m in MODEL:
        dal = np.mean([ap(g.y, g[m]) for _, g in d.groupby("tag")])
        s = ""
        if m != ref:
            lo, hi = boot(d, m, ref)
            s = f"{np.mean([ap(g.y, g[m]) - ar[t] for t, g in d.groupby('tag')]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"  {m:<15}{dal:>7.3f}   {s:<26}{ap(d.y, d[m]):>7.3f}", flush=True)
print("\nselesai.")
