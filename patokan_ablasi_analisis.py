"""
Artikel 2, Percobaan C — tabel ablasi dari DL_FIRE_NASIONAL/patokan_ablasi.csv.

Untuk tiap kelompok k: sumbangan = AUC-PR(lengkap) − AUC-PR(tanpa_k) pada baris yang sama,
selang bootstrap klaster kabupaten 95%. Dilaporkan pada tiga cara pandang, karena Hasil_Patokan_2
menunjukkan hasilnya bisa berbeda:
  - rata-rata dalam-tahun: rata-rata sumbangan per tahun uji S1 2020-2025 (tanpa 2019)
  - tahun ekstrem disisihkan: S2 2015, 2014, 2019, rata-rata dalam-tahun
  - digabung lintas tahun: S1 2020-2025 sebagai satu himpunan uji, cara yang lazim di pustaka.
    Cara ini ikut memberi nilai pada fitur yang hanya membedakan TAHUN, misalnya indeks iklim
    nasional yang bernilai sama untuk semua kabupaten dalam satu bulan.
Sasaran per kabupaten dinilai pada kabupaten dengan ada_positif = 1.

Pakai:  python patokan_ablasi_analisis.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from kelompok_fitur import KELOMPOK

BASE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_csv(os.path.join(BASE, "DL_FIRE_NASIONAL", "patokan_ablasi.csv"), dtype={"tag": str})
TAHUN_BIASA = [str(t) for t in range(2020, 2026)]


def boot_dalam_tahun(d, a, b, n=1000, seed=2026):
    """Rata-rata sumbangan per tahun, bootstrap klaster kabupaten dengan undian sama tiap tahun."""
    uk = d["gid"].unique(); rng = np.random.default_rng(seed)
    per_th = {t: g for t, g in d.groupby("tag")}
    idx = {t: {k: np.where(g["gid"].to_numpy() == k)[0] for k in uk} for t, g in per_th.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, size=len(uk), replace=True); v = []
        for t, g in per_th.items():
            j = np.concatenate([idx[t][k] for k in pilih])
            y = g["y"].to_numpy()[j]
            if y.sum():
                v.append(ap(y, g[a].to_numpy()[j]) - ap(y, g[b].to_numpy()[j]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


def boot_gabung(d, a, b, n=1000, seed=2026):
    g = d["gid"].to_numpy(); uk = np.unique(g); idx = {k: np.where(g == k)[0] for k in uk}
    y, sa, sb = d["y"].to_numpy(), d[a].to_numpy(), d[b].to_numpy()
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(uk, size=len(uk), replace=True)])
        out.append(ap(y[j], sa[j]) - ap(y[j], sb[j]))
    return np.percentile(out, [2.5, 97.5])


def rata_dalam_tahun(d, a, b):
    return np.mean([ap(g["y"], g[a]) - ap(g["y"], g[b]) for _, g in d.groupby("tag") if g["y"].sum()])


for sas in ("y_gabungan", "y_kabupaten"):
    d0 = P[P["sasaran"] == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0["ada_positif"] == 1]
    biasa = d0[(d0["skema"] == "S1") & d0["tag"].isin(TAHUN_BIASA)]
    ekstrem = d0[d0["skema"] == "S2"]
    print("\n" + "=" * 96 + f"\nSASARAN {sas}\n" + "=" * 96)
    print(f"  lengkap vs klimatologi, rata-rata dalam-tahun 2020-25: "
          f"{rata_dalam_tahun(biasa, 'lengkap', 'klimatologi'):+.3f}; tahun ekstrem: "
          f"{rata_dalam_tahun(ekstrem, 'lengkap', 'klimatologi'):+.3f}")
    print(f"\n  {'kelompok dilepas':<18}{'tahun biasa, dalam-tahun':<30}{'tahun ekstrem, dalam-tahun':<30}"
          f"{'2020-25 digabung lintas tahun'}")
    for k in KELOMPOK:
        t = f"tanpa_{k}"
        lo1, hi1 = boot_dalam_tahun(biasa, "lengkap", t)
        lo2, hi2 = boot_dalam_tahun(ekstrem, "lengkap", t)
        lo3, hi3 = boot_gabung(biasa, "lengkap", t)
        print(f"  {k:<18}{rata_dalam_tahun(biasa, 'lengkap', t):+.3f} [{lo1:+.3f}, {hi1:+.3f}]"
              f"{'':<6}{rata_dalam_tahun(ekstrem, 'lengkap', t):+.3f} [{lo2:+.3f}, {hi2:+.3f}]"
              f"{'':<6}{ap(biasa['y'], biasa['lengkap']) - ap(biasa['y'], biasa[t]):+.3f} [{lo3:+.3f}, {hi3:+.3f}]")
print("\nselesai.")
