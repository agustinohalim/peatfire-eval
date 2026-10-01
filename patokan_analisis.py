"""
Artikel 2 (PeatFireBench) — tabel hasil dari DL_FIRE_NASIONAL/patokan_prediksi.csv.

  S1  Tabel utama: AUC-PR dan skill ternormalisasi delapan model, pembagian kronologis 2019-2025,
      dengan selang bootstrap klaster kabupaten untuk selisih terhadap klimatologi.
  S2  Tahun ekstrem disisihkan (2015, 2014, 2019): skill tiap model pada tahun itu, dibandingkan
      dengan skill model yang sama di S1. Untuk 2019, juga dibandingkan dengan S1 tahun 2019 saja
      (tahun yang sama, dilatih hanya pada masa lalu) — memisahkan efek "tahun ekstrem" dari efek
      "disisihkan".
  S3  Transfer antarpulau: AUC-PR model yang dipelajari bila pulau uji tak pernah dilihat, lawan
      model yang sama di S1 (pulau uji terlihat), untuk baris yang sama.

Sasaran utama y_kabupaten dinilai hanya pada kabupaten dengan sedikitnya satu positif
(ada_positif = 1, Kartu_Data_Nasional bagian 3); y_gabungan dinilai pada semua kabupaten.

Pakai:  python patokan_analisis.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

BASE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_csv(os.path.join(BASE, "DL_FIRE_NASIONAL", "patokan_prediksi.csv"), dtype={"tag": str})
MODEL = ["klimatologi", "persistence", "seasonal_naive", "rasio", "logistik", "rf", "xgb", "mlp"]
NAMA = {"klimatologi": "Climatology", "persistence": "Persistence", "seasonal_naive": "Seasonal naive",
        "rasio": "Ratio scaling", "logistik": "Logistic (ONI, DMI)", "rf": "Random forest",
        "xgb": "Gradient boosting", "mlp": "MLP"}


def skill(y, s):
    p = y.mean()
    return (ap(y, s) - p) / (1 - p)


def boot_beda(d, a, b, n=1000, seed=2026):
    g = d["gid"].to_numpy(); uk = np.unique(g); idx = {k: np.where(g == k)[0] for k in uk}
    y, sa, sb = d["y"].to_numpy(), d[a].to_numpy(), d[b].to_numpy()
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(uk, size=len(uk), replace=True)])
        if y[j].sum():
            out.append(ap(y[j], sa[j]) - ap(y[j], sb[j]))
    return np.percentile(out, [2.5, 97.5])


for sas in ("y_kabupaten", "y_gabungan"):
    d0 = P[P["sasaran"] == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0["ada_positif"] == 1]
    print("\n" + "=" * 100)
    print(f"SASARAN {sas}  ({d0['gid'].nunique()} kabupaten dinilai)")
    print("=" * 100)

    s1 = d0[d0["skema"] == "S1"]
    print(f"\nS1 kronologis 2019-2025: {len(s1)} baris, {int(s1['y'].sum())} positif, "
          f"prevalensi {s1['y'].mean() * 100:.2f}%")
    print(f"  {'model':<22}{'AUC-PR':>8}{'skill':>8}   {'selisih vs klimatologi [95% klaster]':<40}")
    skill_s1 = {}
    for m in sorted(MODEL, key=lambda m: -ap(s1["y"], s1[m])):
        a = ap(s1["y"], s1[m]); sk = skill(s1["y"], s1[m]); skill_s1[m] = sk
        teks = ""
        if m != "klimatologi":
            lo, hi = boot_beda(s1, m, "klimatologi")
            teks = f"{a - ap(s1['y'], s1['klimatologi']):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"  {NAMA[m]:<22}{a:>8.3f}{sk:>8.3f}   {teks}")

    print(f"\nS2 tahun ekstrem disisihkan: skill ternormalisasi (S1 gabungan sebagai acuan)")
    print(f"  {'model':<22}{'S1':>7}" + "".join(f"{t:>9}" for t in ("2015", "2014", "2019")) + f"{'S1 2019 saja':>14}")
    s2 = d0[d0["skema"] == "S2"]
    s1_19 = s1[s1["tahun"] == 2019]
    for m in MODEL:
        v = [skill(s2[s2["tag"] == t]["y"], s2[s2["tag"] == t][m]) for t in ("2015", "2014", "2019")]
        print(f"  {NAMA[m]:<22}{skill_s1[m]:>7.3f}" + "".join(f"{x:>9.3f}" for x in v)
              + f"{skill(s1_19['y'], s1_19[m]):>14.3f}")
    for t in ("2015", "2014", "2019"):
        dd = s2[s2["tag"] == t]
        print(f"  {t}: {int(dd['y'].sum())} positif, prevalensi {dd['y'].mean() * 100:.1f}%")
    # Skill ternormalisasi tidak kebal prevalensi, jadi antartahun dibandingkan lewat keunggulan
    # atas klimatologi pada baris yang sama (AUC-PR, selisih dalam tahun itu).
    print("  keunggulan AUC-PR gradient boosting atas klimatologi, baris sama [95% klaster]:")
    for t, dd in [("S1", s1)] + [(t, s2[s2["tag"] == t]) for t in ("2015", "2014", "2019")]:
        lo, hi = boot_beda(dd, "xgb", "klimatologi")
        print(f"    {t:<5}{ap(dd['y'], dd['xgb']) - ap(dd['y'], dd['klimatologi']):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print(f"\nS3 transfer antarpulau: AUC-PR model dipelajari, pulau tak terlihat (S3) lawan terlihat (S1)")
    s3 = d0[d0["skema"] == "S3"]
    print(f"  {'pulau':<14}{'pos':>5}" + "".join(f"{NAMA[m][:10]:>12}" for m in ("xgb", "rf", "mlp", "logistik"))
          + f"{'klim':>8}{'xgb S1':>8}{'rugi xgb':>10}")
    for g in sorted(s3["pulau"].unique()):
        a3 = s3[s3["pulau"] == g]; a1 = s1[s1["pulau"] == g]
        if a3["y"].sum() < 5:
            print(f"  {g:<14}{int(a3['y'].sum()):>5}  terlalu sedikit positif"); continue
        vals = "".join(f"{ap(a3['y'], a3[m]):>12.3f}" for m in ("xgb", "rf", "mlp", "logistik"))
        x1 = ap(a1["y"], a1["xgb"])
        print(f"  {g:<14}{int(a3['y'].sum()):>5}{vals}{ap(a3['y'], a3['klimatologi']):>8.3f}{x1:>8.3f}"
              f"{x1 - ap(a3['y'], a3['xgb']):>+10.3f}")

    # Selang klaster untuk dua pembandingan yang memakai baris sama:
    #  (a) S2 2019 lawan S1 2019: tahun sama, dilatih dengan atau tanpa tahun sesudahnya;
    #  (b) S3 lawan S1 per pulau: model dilatih tanpa atau dengan pulau uji.
    kunci = ["gid", "tahun", "bulan"]
    print("\nSelang klaster 95% (baris dipasangkan): S2 2019 - S1 2019, dan S1 - S3 (rugi transfer)")
    a = s2[s2["tag"] == "2019"][kunci + ["y"] + MODEL].merge(
        s1_19[kunci + MODEL], on=kunci, suffixes=("_s2", "_s1"))
    for m in ("xgb", "rf", "mlp"):
        lo, hi = boot_beda(a, f"{m}_s2", f"{m}_s1")
        print(f"  2019 {NAMA[m]:<20}{ap(a['y'], a[m + '_s2']) - ap(a['y'], a[m + '_s1']):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for g in sorted(s3["pulau"].unique()):
        b = s1[s1["pulau"] == g][kunci + ["y", "xgb"]].merge(
            s3[s3["pulau"] == g][kunci + ["xgb"]], on=kunci, suffixes=("_s1", "_s3"))
        if b["y"].sum() < 5:
            continue
        lo, hi = boot_beda(b, "xgb_s1", "xgb_s3")
        print(f"  S3 {g:<14} xgb {ap(b['y'], b['xgb_s1']) - ap(b['y'], b['xgb_s3']):+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("\nselesai.")
