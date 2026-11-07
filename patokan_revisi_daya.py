"""
PeatFireBench revisi fase G1 (Hasil_Patokan_12): daya uji skor Brier atas EF-L.

Membaca BSS per tahun atas EF-L dari log fase E (`patokan_revisi_peluang_{kab,nas}.log`, blok "-- atas
ef1_tingkat": rf, xgb, xgb_setel) dan fase F (`patokan_revisi_tingkat_{kab,nas}.log`: xgb_nas), lalu
menghitung selisih minimum terdeteksi pada daya 80 % dengan uji t dua sisi 5 % atas enam tahun,
(t_0,975 + t_0,80; df n-1) * sd / sqrt(n), sama dengan rumus AUC-PR fase D, dan banyaknya tahun yang
dibutuhkan untuk mendeteksi rata-rata yang teramati (iterasi df).

Pakai:    python patokan_revisi_daya.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_daya.log
"""
import os
import re

import numpy as np
from scipy import stats

L = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL", "penuh")
POLA = re.compile(r"per tahun ((?:[+-]\d+\.\d+\s*){6})")


def tahunan_e(berkas, model):
    t = open(os.path.join(L, berkas), encoding="utf-8").read()
    blok = t[t.index("-- atas ef1_tingkat"):]
    for baris in blok.splitlines()[1:]:
        if baris.split() and baris.split()[0] == model:
            return np.array(POLA.search(baris).group(1).split(), float)
    raise KeyError(model)


def tahunan_f(berkas):
    for baris in open(os.path.join(L, berkas), encoding="utf-8"):
        if "BSS xgb_nas atas ef1_tingkat" in baris:
            return np.array(POLA.search(baris).group(1).split(), float)
    raise KeyError("xgb_nas")


def mdd(x):
    n = len(x); sd = x.std(ddof=1)
    return (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n)


def tahun_perlu(x):
    m, sd = abs(x.mean()), x.std(ddof=1)
    if m == 0:
        return np.inf
    for n in range(3, 1000):
        if (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n) <= m:
            return n
    return np.inf


print("G1. Daya uji BSS atas EF-L, tahun biasa S1 2020-2025 (n = 6, df 5)")
for sas, e, f in (("y_kabupaten", "patokan_revisi_peluang_kab.log", "patokan_revisi_tingkat_kab.log"),
                  ("y_gabungan", "patokan_revisi_peluang_nas.log", "patokan_revisi_tingkat_nas.log")):
    print(f"\n--- {sas}")
    for nama, x in (("rf", tahunan_e(e, "rf")), ("xgb", tahunan_e(e, "xgb")),
                    ("xgb_setel", tahunan_e(e, "xgb_setel")), ("xgb_nas", tahunan_f(f))):
        print(f"    {nama:<10} rata-rata {x.mean():+.4f}  sd {x.std(ddof=1):.4f}  MDD80 {mdd(x):.4f}  "
              f"tahun dibutuhkan {tahun_perlu(x)}  per tahun " + " ".join(f"{v:+.3f}" for v in x))
print("\nselesai.")
