"""
Artikel 2 — apakah curah hujan CHIRPS mengubah hasil tolok ukur? Membandingkan
patokan_prediksi.csv (tanpa hujan) dengan patokan_prediksi_hujan.csv (dengan jeda hujan) pada
baris yang sama.

  1. Keunggulan gradient boosting atas klimatologi per tahun uji, dengan dan tanpa hujan
     (mengulang Hasil_Patokan_2 bagian 1): apakah pola berlawanan dua sasaran bertahan?
  2. Sumbangan hujan: AUC-PR xgb/rf/mlp dengan hujan − tanpa hujan, selang klaster 95%,
     untuk S1 non-ekstrem 2020-2025, tiap tahun ekstrem disisihkan, dan transfer antarpulau.
  3. ECE gradient boosting dengan dan tanpa hujan, non-ekstrem lawan ekstrem.

Pakai:  python patokan_banding_hujan.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
KUNCI = ["sasaran", "skema", "tag", "gid", "tahun", "bulan"]
# Bawaan: tanpa hujan (A) lawan dengan hujan (B). BANDING_A / BANDING_B mengganti berkasnya,
# misalnya BANDING_A=patokan_prediksi_hujan.csv BANDING_B=patokan_prediksi_lengkap.csv untuk
# sumbangan lapisan lahan dan manusia. Label "tanpa/dengan hujan" di keluaran lalu berarti A/B.
A = pd.read_csv(os.path.join(D, os.environ.get("BANDING_A", "patokan_prediksi.csv")), dtype={"tag": str})
B = pd.read_csv(os.path.join(D, os.environ.get("BANDING_B", "patokan_prediksi_hujan.csv")), dtype={"tag": str})
M = A.merge(B[KUNCI + ["xgb", "rf", "mlp", "klimatologi"]], on=KUNCI, suffixes=("", "_h"),
            validate="one_to_one")
assert len(M) == len(A) == len(B)
assert np.allclose(M["klimatologi"], M["klimatologi_h"]), "baseline lokal mestinya identik"


def boot_beda(d, a, b, n=1000, seed=2026):
    g = d["gid"].to_numpy(); uk = np.unique(g); idx = {k: np.where(g == k)[0] for k in uk}
    y, sa, sb = d["y"].to_numpy(), d[a].to_numpy(), d[b].to_numpy()
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(uk, size=len(uk), replace=True)])
        if y[j].sum():
            out.append(ap(y[j], sa[j]) - ap(y[j], sb[j]))
    return np.percentile(out, [2.5, 97.5])


def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())


def teks(d, a, b):
    lo, hi = boot_beda(d, a, b)
    return f"{ap(d['y'], d[a]) - ap(d['y'], d[b]):+.3f} [{lo:+.3f}, {hi:+.3f}]"


for sas in ("y_gabungan", "y_kabupaten"):
    d0 = M[M["sasaran"] == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0["ada_positif"] == 1]
    s1, s2, s3 = (d0[d0["skema"] == k] for k in ("S1", "S2", "S3"))
    print("\n" + "=" * 90 + f"\nSASARAN {sas}\n" + "=" * 90)

    print("\n1. Keunggulan xgb atas klimatologi per tahun: tanpa hujan | dengan hujan")
    grup = [(f"S1 {t}", s1[s1["tag"] == str(t)]) for t in range(2019, 2026)]
    grup.append(("S1 non-ekstrem 20-25", s1[s1["tahun"] >= 2020]))
    grup += [(f"S2 {t}", s2[s2["tag"] == t]) for t in ("2015", "2014", "2019")]
    for nama, dd in grup:
        print(f"  {nama:<22}{teks(dd, 'xgb', 'klimatologi'):<28}{teks(dd, 'xgb_h', 'klimatologi')}")

    print("\n2. Sumbangan hujan: AUC-PR dengan hujan - tanpa hujan [95% klaster]")
    grup2 = [("S1 non-ekstrem 20-25", s1[s1["tahun"] >= 2020])]
    grup2 += [(f"S2 {t}", s2[s2["tag"] == t]) for t in ("2015", "2014", "2019")]
    grup2 += [(f"S3 {g}", s3[s3["pulau"] == g]) for g in sorted(s3["pulau"].unique())]
    print(f"  {'uji':<22}" + "".join(f"{m:<28}" for m in ("xgb", "rf", "mlp")))
    for nama, dd in grup2:
        print(f"  {nama:<22}" + "".join(f"{teks(dd, m + '_h', m):<28}" for m in ("xgb", "rf", "mlp")))

    print("\n3. ECE gradient boosting: tanpa hujan | dengan hujan")
    for nama, dd in [("S1 non-ekstrem 20-25", s1[s1["tahun"] >= 2020]), ("S2 ekstrem gabungan", s2)]:
        y = dd["y"].to_numpy()
        print(f"  {nama:<22}{ece(y, dd['xgb'].to_numpy()):.3f} | {ece(y, dd['xgb_h'].to_numpy()):.3f}")
print("\nselesai.")
