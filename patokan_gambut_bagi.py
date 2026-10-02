"""
PeatFireBench — hasil utama dipisah menurut gambut: kabupaten bergambut (gambut_fraksi > 0,
131 satuan) lawan tanpa gambut. Menjawab apakah temuan bertahan di tempat gambut terbakar.

Untuk tiap kelompok dan sasaran: keunggulan dalam-tahun atas klimatologi (S1 2020-2025 dan S2
2015/2014/2019) untuk gradient boosting, random forest, MLP, persistence, dan dua baseline FWI
(skala nasional; klimatologi + FWI dari patokan_fwi_adil.csv); juga kontras ekstrem - tenang
gradient boosting sebagai bagian ruang sisa (temuan 1) dan selisih AUC-PR digabung-lintas-tahun
dikurangi dalam-tahun (temuan 2). Selang 95 % bootstrap kabupaten (500 ulangan).

Pakai:    python patokan_gambut_bagi.py   (DL_FIRE_NASIONAL/penuh/patokan_gambut_bagi.log)
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from kelompok_fitur import PREDIKSI

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
k = ["gid", "tahun", "bulan", "sasaran", "skema", "tag"]
P = pd.read_csv(os.path.join(D, PREDIKSI), dtype={"tag": str})
P = P[P.skema.isin(["S1", "S2"])]
F = pd.read_csv(os.path.join(D, "patokan_fwi_adil.csv"), dtype={"tag": str})[k + ["klim_fwi", "klim_saja"]]
P = P.merge(F, on=k, how="left", validate="one_to_one")
g = pd.read_csv(os.path.join(D, "gambut_kabupaten.csv")).set_index("gid")["gambut_fraksi"]
P["gambut"] = P.gid.map(g).fillna(0) > 0
BIASA = [str(t) for t in range(2020, 2026)]
EKS = ["2015", "2014", "2019"]
rng = np.random.default_rng(2026)
MODEL = ["xgb", "rf", "mlp", "persistence", "fwi", "klim_fwi"]


def per_tahun(d, a, b, ruang=False):
    v = []
    for _, x in d.groupby("tag"):
        if x.y.sum():
            pa, pb = ap(x.y, x[a]), ap(x.y, x[b])
            v.append((pa - pb) / (1 - pb) if ruang else pa - pb)
    return np.mean(v)


def boot(fn, d, n=500):
    uk = d.gid.unique(); idx = {u: np.where(d.gid.to_numpy() == u)[0] for u in uk}
    out = []
    for _ in range(n):
        j = np.concatenate([idx[u] for u in rng.choice(uk, len(uk))])
        out.append(fn(d.iloc[j]))
    return np.nanpercentile(out, [2.5, 97.5])


for sas in ("y_kabupaten", "y_gabungan"):
    d0 = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    for nama, grup in (("bergambut", True), ("tanpa gambut", False)):
        d = d0[d0.gambut == grup]
        b, e = d[(d.skema == "S1") & d.tag.isin(BIASA)], d[d.skema == "S2"]
        print(f"\n=== {sas}, {nama}: {d.gid.nunique()} satuan; positif S1 2020-25 {int(b.y.sum())}, "
              f"S2 {int(e.y.sum())}; klim AUC-PR S1 {np.mean([ap(x.y, x.klimatologi) for _, x in b.groupby('tag') if x.y.sum()]):.3f}")
        for m in MODEL:
            lo, hi = boot(lambda x: per_tahun(x, m, "klimatologi"), b)
            lo2, hi2 = boot(lambda x: per_tahun(x, m, "klimatologi"), e)
            print(f"  {m:12s} S1 {per_tahun(b, m, 'klimatologi'):+.3f} [{lo:+.3f}, {hi:+.3f}]   "
                  f"S2 {per_tahun(e, m, 'klimatologi'):+.3f} [{lo2:+.3f}, {hi2:+.3f}]")
        fw = lambda x: per_tahun(x, "klim_fwi", "klim_saja")
        lo, hi = boot(fw, b)
        print(f"  FWI di atas tempat (klim_fwi - klim_saja), S1: {fw(b):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        kont = lambda x: per_tahun(x[x.skema == "S2"], "xgb", "klimatologi", True) - \
            per_tahun(x[(x.skema == "S1") & x.tag.isin(BIASA)], "xgb", "klimatologi", True)
        lo, hi = boot(kont, d)
        print(f"  temuan 1: kontras ekstrem - tenang GB, bagian ruang sisa: {kont(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        gab = lambda x: (ap(x.y, x.xgb) - ap(x.y, x.klimatologi)) - per_tahun(x, "xgb", "klimatologi")
        lo, hi = boot(gab, b)
        print(f"  temuan 2: keunggulan GB digabung - dalam-tahun, S1 2020-25: {gab(b):+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("\nselesai.")
