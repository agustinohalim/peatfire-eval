"""
Percobaan 6 — apakah temuan bertahan bila kabupaten dipisahkan?

Keberatan (tinjauan mandiri, butir 6): dengan satu ambang gabungan (> 369 titik), sebagian
besar tugas menjadi membedakan Ketapang dari kabupaten kecil. Itu bisa menjelaskan mengapa
klimatologi mencapai ROC-AUC 0,95.

  A. Antar lawan dalam kabupaten. Prediktor paling bodoh — rerata titik panas kabupaten di
     data latih, tanpa musim — diukur pada ambang gabungan. Bila ia sudah tinggi, sebagian
     besar skill gabungan adalah skill antar-kabupaten.
  B. Metrik bertingkat kabupaten, ambang gabungan. ROC-AUC dan AUC-PR dihitung di dalam tiap
     kabupaten yang punya positif dan negatif di data uji, lalu dirata-rata.
  C. Ambang per kabupaten. Label = melebihi persentil 90 kabupaten itu sendiri (2012-2025),
     jadi tiap kabupaten punya prevalensi sekitar 10%. Seluruh percobaan 2x2 diulang: peringkat
     empat sel, tau, pembalikan atas 300 undian, dan selang bootstrap selisih GB - klimatologi.

Tidak menyunting naskah.

Pakai:
    python percobaan6_kabupaten.py DL_FIRE_SV-C2_792597/panel_bulanan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "percobaan.py"),
          encoding="utf-8").read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
df, ambang = muat(panel_path, oni_path)
TAHUN_UJI = list(range(2019, 2026))
GB, KL = "gradient-boosting", "klimatologi"


def lipat(d):
    """Rolling-origin; kembalikan y, kabupaten, dan skor mentah per model untuk baris uji."""
    ys, kb, sk = [], [], None
    for th in TAHUN_UJI:
        latih, uji = d[d["tahun"] < th], d[d["tahun"] == th]
        s = skor_model(latih, uji)
        if sk is None:
            sk = {n: [] for n in s}
        for n, (_, su) in s.items():
            sk[n].append(su)
        ys.append(uji["y"].to_numpy())
        kb.append(uji["kabupaten"].to_numpy())
    return np.concatenate(ys), np.concatenate(kb), {n: np.concatenate(v) for n, v in sk.items()}


def peringkat(nilai):
    urut = sorted(nilai, key=lambda n: -nilai[n])
    return {n: urut.index(n) + 1 for n in nilai}


def tau(r1, r2):
    k = list(r1)
    return kendalltau([r1[n] for n in k], [r2[n] for n in k]).statistic


def balik(r1, r2):
    k = list(r1)
    return sum(1 for i in range(len(k)) for j in range(i + 1, len(k))
               if (r1[k[i]] - r1[k[j]]) * (r2[k[i]] - r2[k[j]]) < 0)


def empat_sel(y, sk, judul):
    nama = list(sk)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(7)
    D = {n: average_precision_score(y, sk[n]) for n in nama}
    C = {n: roc_auc_score(y, sk[n]) for n in nama}
    rD, rC = peringkat(D), peringkat(C)
    rA_list, balik_list, gb_atas = [], [], 0
    A_sum = {n: 0.0 for n in nama}
    for _ in range(300):
        i = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        A = {n: roc_auc_score(y[i], sk[n][i]) for n in nama}
        for n in nama:
            A_sum[n] += A[n] / 300
        rA = peringkat(A)
        rA_list.append(rA)
        balik_list.append(balik(rA, rD))
        gb_atas += A[GB] > A[KL]
    rA_rerata = peringkat({n: -np.mean([r[n] for r in rA_list]) for n in nama})
    print(f"\n{judul}")
    print(f"  n uji {len(y)}, positif {int(y.sum())}, prevalensi {y.mean() * 100:.1f}%")
    print(f"  {'model':<20}{'ROC-AUC seimb.':>15}{'ROC-AUC asli':>13}{'AUC-PR':>8}   rank A  C  D")
    for n in sorted(nama, key=lambda n: rD[n]):
        print(f"  {n:<20}{A_sum[n]:>15.3f}{C[n]:>13.3f}{D[n]:>8.3f}      {rA_rerata[n]}  {rC[n]}  {rD[n]}")
    b = np.array(balik_list)
    print(f"  tau C vs D (metrik saja) = {tau(rC, rD):+.3f}; pembalikan cell A vs D: median "
          f"{int(np.median(b))}, rentang {b.min()}-{b.max()}, nol pada {(b == 0).sum()}/300")
    print(f"  GB di atas klimatologi menurut ROC-AUC seimbang: {gb_atas}/300 undian")
    rb = np.random.default_rng(2026)
    beda = []
    for _ in range(2000):
        i = rb.integers(0, len(y), len(y))
        beda.append(average_precision_score(y[i], sk[GB][i]) - average_precision_score(y[i], sk[KL][i]))
    lo, hi = np.percentile(beda, [2.5, 97.5])
    print(f"  selisih AUC-PR GB - klimatologi {D[GB] - D[KL]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return rD


# ============================================================ A. antar kabupaten

print("=" * 78)
print("A. SEBERAPA BESAR SKILL GABUNGAN BERASAL DARI MEMBEDAKAN KABUPATEN")
print("=" * 78)
y, kb, sk = lipat(df)
rerata_kab = []
for th in TAHUN_UJI:
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    m = latih.groupby("kabupaten", observed=True)["titik_panas"].mean()
    rerata_kab.append(uji["kabupaten"].map(m).fillna(0).to_numpy())
rerata_kab = np.concatenate(rerata_kab)
print(f"  prediktor 'rerata kabupaten' (tanpa musim): ROC-AUC {roc_auc_score(y, rerata_kab):.3f}, "
      f"AUC-PR {average_precision_score(y, rerata_kab):.3f}")
print(f"  klimatologi (kabupaten x bulan)           : ROC-AUC {roc_auc_score(y, sk[KL]):.3f}, "
      f"AUC-PR {average_precision_score(y, sk[KL]):.3f}")
per_kab = pd.Series(y).groupby(kb).agg(["sum", "count"])
per_kab.columns = ["positif", "baris"]
print("\n  positif per kabupaten pada data uji (ambang gabungan > 369):")
for k, r in per_kab.sort_values("positif", ascending=False).iterrows():
    print(f"    {k:<22}{int(r['positif']):>4} dari {int(r['baris'])}")

# ============================================================ B. bertingkat

print("\n" + "=" * 78)
print("B. METRIK DI DALAM KABUPATEN, AMBANG GABUNGAN")
print("=" * 78)
layak = [k for k in np.unique(kb) if 0 < y[kb == k].sum() < (kb == k).sum()]
print(f"  kabupaten dengan positif dan negatif di data uji: {len(layak)} dari {len(np.unique(kb))}")
print(f"  {'model':<20}{'ROC-AUC rerata':>15}{'AUC-PR rerata':>15}{'skill rerata':>14}")
tabB = {}
for n in sk:
    roc = [roc_auc_score(y[kb == k], sk[n][kb == k]) for k in layak]
    ap = [average_precision_score(y[kb == k], sk[n][kb == k]) for k in layak]
    skl = [(a - y[kb == k].mean()) / (1 - y[kb == k].mean()) for a, k in zip(ap, layak)]
    tabB[n] = (np.mean(roc), np.mean(ap), np.mean(skl))
for n in sorted(tabB, key=lambda n: -tabB[n][1]):
    print(f"  {n:<20}{tabB[n][0]:>15.3f}{tabB[n][1]:>15.3f}{tabB[n][2]:>14.3f}")
rR = peringkat({n: tabB[n][0] for n in tabB})
rP = peringkat({n: tabB[n][1] for n in tabB})
print(f"  tau ROC-AUC vs AUC-PR (di dalam kabupaten) = {tau(rR, rP):+.3f}, pembalikan {balik(rR, rP)} dari 21")

# ============================================================ C. ambang per kabupaten

print("\n" + "=" * 78)
print("C. AMBANG PER KABUPATEN (persentil 90 tiap kabupaten, 2012-2025)")
print("=" * 78)
rD_gab = empat_sel(y, sk, "  [pembanding] ambang gabungan > 369")

mentah = pd.read_csv(panel_path)
mentah = mentah[mentah["tahun"] <= 2025]
q_kab = mentah.groupby("kabupaten")["titik_panas"].quantile(0.90)
d2 = df.copy()
d2["y"] = (d2["titik_panas"] > d2["kabupaten"].map(q_kab)).astype(int)
print("\n  ambang per kabupaten: " + ", ".join(f"{k.split()[-1]} {v:.0f}" for k, v in q_kab.sort_values().items()))
y2, kb2, sk2 = lipat(d2)
rD_kab = empat_sel(y2, sk2, "  ambang per kabupaten")
print(f"\n  tau peringkat AUC-PR gabungan vs per kabupaten = {tau(rD_gab, rD_kab):+.3f}")

print("\nselesai.")
