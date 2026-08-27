"""
Percobaan 4: memisahkan dua faktor yang selama ini dibaurkan.

Percobaan 1 membandingkan dua protokol yang berbeda dalam DUA hal sekaligus:
penyeimbangan data uji dan pilihan metrik. Karena itu naskah belum dapat menjawab
pertanyaan reviewer yang paling berbahaya: pembalikan peringkat itu disebabkan
penyeimbangan, atau metriknya?

Rancangan 2x2:

                      ROC-AUC          AUC-PR
    seimbang          sel A            sel B
    prevalensi asli   sel C            sel D

  sel A = protokol lazim        sel D = protokol operasional
  sel B dan C = sel silang yang memisahkan kedua faktor

Kesepakatan peringkat diukur dengan Kendall tau, bukan hitungan pasangan buatan
sendiri, dan dilaporkan dengan selang persentil dari 300 undian penyeimbangan.

Pakai:
    python percobaan4_matriks2x2.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \\
        DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
UNDIAN = 300
rng = np.random.default_rng(11)

# ------------------------------------------------------------------ kumpulkan skor

df, ambang = muat(panel_path, oni_path)
kum = {}
for th in range(2019, 2026):
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    for nama, (sl, su) in skor_model(latih, uji).items():
        kum.setdefault(nama, {"y": [], "raw": []})
        kum[nama]["y"].append(uji["y"].to_numpy())
        kum[nama]["raw"].append(su)

NAMA = list(kum)
y = np.concatenate(kum[NAMA[0]]["y"])
raw = {n: np.concatenate(kum[n]["raw"]) for n in NAMA}
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]

print(f"Panel: {len(df)} baris, ambang > {ambang:.0f}, "
      f"prevalensi uji {y.mean() * 100:.1f}%, {len(NAMA)} model, {UNDIAN} undian")

# ------------------------------------------------------------------ empat sel


def peringkat(nilai):
    """Peringkat 1 = terbaik. nilai: dict nama -> skor, makin besar makin baik."""
    urut = sorted(NAMA, key=lambda n: -nilai[n])
    return {n: urut.index(n) + 1 for n in NAMA}


# sel C dan D: prevalensi asli, tidak ada undian, jadi tetap
selC_nilai = {n: roc_auc_score(y, raw[n]) for n in NAMA}
selD_nilai = {n: average_precision_score(y, raw[n]) for n in NAMA}
selC, selD = peringkat(selC_nilai), peringkat(selD_nilai)

# sel A dan B: data uji diseimbangkan, satu peringkat per undian
selA_undian, selB_undian = [], []
selA_nilai_kum = {n: [] for n in NAMA}
selB_nilai_kum = {n: [] for n in NAMA}
for _ in range(UNDIAN):
    idx = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
    yb = y[idx]
    a = {n: roc_auc_score(yb, raw[n][idx]) for n in NAMA}
    b = {n: average_precision_score(yb, raw[n][idx]) for n in NAMA}
    for n in NAMA:
        selA_nilai_kum[n].append(a[n])
        selB_nilai_kum[n].append(b[n])
    selA_undian.append(peringkat(a))
    selB_undian.append(peringkat(b))

selA_rerata = {n: float(np.mean(selA_nilai_kum[n])) for n in NAMA}
selB_rerata = {n: float(np.mean(selB_nilai_kum[n])) for n in NAMA}
selA, selB = peringkat(selA_rerata), peringkat(selB_rerata)

print("\n" + "=" * 78)
print("A. EMPAT SEL — peringkat model (1 = terbaik)")
print("=" * 78)
print(f"{'Model':<20}{'A seimbang':>12}{'B seimbang':>12}{'C asli':>10}{'D asli':>10}")
print(f"{'':<20}{'ROC-AUC':>12}{'AUC-PR':>12}{'ROC-AUC':>10}{'AUC-PR':>10}")
print("-" * 78)
for n in sorted(NAMA, key=lambda n: selA[n]):
    print(f"{n:<20}{selA[n]:>12}{selB[n]:>12}{selC[n]:>10}{selD[n]:>10}")
print("-" * 78)
print(f"{'nilai metrik':<20}")
for n in sorted(NAMA, key=lambda n: selA[n]):
    print(f"{n:<20}{selA_rerata[n]:>12.3f}{selB_rerata[n]:>12.3f}"
          f"{selC_nilai[n]:>10.3f}{selD_nilai[n]:>10.3f}")

# ------------------------------------------------------------------ penguraian


def tau_tetap(p1, p2):
    v1 = [p1[n] for n in NAMA]
    v2 = [p2[n] for n in NAMA]
    return kendalltau(v1, v2).statistic


def tau_sebaran(undian, tetap):
    v = [kendalltau([p[n] for n in NAMA], [tetap[n] for n in NAMA]).statistic
         for p in undian]
    return np.array(v)


def lapor(nama, arr):
    print(f"  {nama:<52} tau = {np.median(arr):+.3f}  "
          f"[{np.percentile(arr, 2.5):+.3f}, {np.percentile(arr, 97.5):+.3f}]")


print("\n" + "=" * 78)
print("B. PENGURAIAN — faktor mana yang membalik peringkat?")
print("=" * 78)

print("\nEfek PENYEIMBANGAN, metrik ditahan tetap:")
lapor("ROC-AUC: seimbang (A) lawan prevalensi asli (C)", tau_sebaran(selA_undian, selC))
lapor("AUC-PR : seimbang (B) lawan prevalensi asli (D)", tau_sebaran(selB_undian, selD))

print("\nEfek METRIK, penyeimbangan ditahan tetap:")
tau_AB = np.array([kendalltau([a[n] for n in NAMA], [b[n] for n in NAMA]).statistic
                   for a, b in zip(selA_undian, selB_undian)])
lapor("pada data seimbang: ROC-AUC (A) lawan AUC-PR (B)", tau_AB)
print(f"  {'pada prevalensi asli: ROC-AUC (C) lawan AUC-PR (D)':<52} "
      f"tau = {tau_tetap(selC, selD):+.3f}  [tetap, tanpa undian]")

print("\nPerbandingan protokol utuh, kedua faktor sekaligus:")
lapor("protokol lazim (A) lawan protokol operasional (D)", tau_sebaran(selA_undian, selD))

# ------------------------------------------------------------------ kesimpulan angka

ef_seimbang = np.median(tau_sebaran(selA_undian, selC))
ef_metrik = tau_tetap(selC, selD)
print("\n" + "=" * 78)
print("C. RINGKASAN")
print("=" * 78)
print(f"  menahan metrik, mengubah penyeimbangan : tau = {ef_seimbang:+.3f}")
print(f"  menahan penyeimbangan, mengubah metrik : tau = {ef_metrik:+.3f}")
if ef_seimbang > ef_metrik:
    print("\n  Metrik adalah faktor yang menata ulang peringkat.")
    print("  Penyeimbangan tidak menata ulang secara sistematis; ia menambah ragam,")
    print("  sehingga satu peringkat yang dilaporkan tidak dapat diulang.")
else:
    print("\n  Penyeimbangan adalah faktor yang menata ulang peringkat.")

sd_A = {n: float(np.std(selA_nilai_kum[n])) for n in NAMA}
print(f"\n  ragam akibat penyeimbangan, sd ROC-AUC antar undian: "
      f"{min(sd_A.values()):.4f} s/d {max(sd_A.values()):.4f}")
pindah = {n: (min(p[n] for p in selA_undian), max(p[n] for p in selA_undian)) for n in NAMA}
print("  rentang peringkat sel A antar undian:")
for n in sorted(NAMA, key=lambda n: selA[n]):
    lo, hi = pindah[n]
    print(f"    {n:<20} {lo}–{hi}" + ("" if lo == hi else "   berpindah"))
