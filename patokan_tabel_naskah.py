"""
Artikel 2 — tabel naskah dari putaran panel lengkap (DL_FIRE_NASIONAL/patokan_prediksi_lengkap.csv).

Tabel 3 (utama): delapan model, dua sasaran; AUC-PR rata-rata dalam-tahun 2020-2025,
                 keunggulan dalam-tahun atas klimatologi [95% klaster], AUC-PR digabung
                 2020-2025, dan ECE (hanya model berkeluaran peluang).
Tabel 4        : tahun ekstrem disisihkan (S2): keunggulan dalam-tahun gradient boosting,
                 random forest, dan MLP atas klimatologi per tahun.
Tabel 5        : transfer antarpulau (S3): rugi AUC-PR gradient boosting (S1 − S3) per kelompok
                 pulau, baris sama, rata-rata dalam-tahun.
Ringkasan panel: jumlah baris, satuan, positif, prevalensi per sasaran.

Keluaran: cetak ke layar (simpan ke DL_FIRE_NASIONAL/patokan_tabel_naskah.log).
Pakai:    python patokan_tabel_naskah.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
# PATOKAN_BERKAS=patokan_prediksi_kepekaan.csv untuk uji kepekaan ambang (tanpa S3)
P = pd.read_csv(os.path.join(D, os.environ.get("PATOKAN_BERKAS", "patokan_prediksi_lengkap.csv")), dtype={"tag": str})
MODEL = ["klimatologi", "persistence", "seasonal_naive", "rasio", "logistik", "rf", "xgb", "mlp"]
NAMA = {"klimatologi": "Climatology", "persistence": "Persistence", "seasonal_naive": "Seasonal naive",
        "rasio": "Ratio scaling", "logistik": "Logistic (ONI, DMI)", "rf": "Random forest",
        "xgb": "Gradient boosting", "mlp": "MLP"}
PELUANG = {"logistik", "rf", "xgb", "mlp"}
BIASA = [str(t) for t in range(2020, 2026)]


def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())


def dalam_tahun(d, a, b=None):
    v = []
    for _, g in d.groupby("tag"):
        if g["y"].sum():
            v.append(ap(g["y"], g[a]) - (ap(g["y"], g[b]) if b else 0.0))
    return float(np.mean(v))


def boot_dalam_tahun(d, a, b, n=1000, seed=2026):
    uk = d["gid"].unique(); rng = np.random.default_rng(seed)
    per = {t: g for t, g in d.groupby("tag")}
    idx = {t: {k: np.where(g["gid"].to_numpy() == k)[0] for k in uk} for t, g in per.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, size=len(uk), replace=True); v = []
        for t, g in per.items():
            j = np.concatenate([idx[t][k] for k in pilih]); y = g["y"].to_numpy()[j]
            if y.sum():
                v.append(ap(y, g[a].to_numpy()[j]) - ap(y, g[b].to_numpy()[j]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


for sas in ("y_kabupaten", "y_gabungan"):
    d0 = P[P["sasaran"] == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0["ada_positif"] == 1]
    s1 = d0[d0["skema"] == "S1"]; biasa = s1[s1["tag"].isin(BIASA)]
    s2 = d0[d0["skema"] == "S2"]; s3 = d0[d0["skema"] == "S3"]
    print("\n" + "=" * 100 + f"\nSASARAN {sas}: {d0['gid'].nunique()} satuan dinilai; S1 2020-25 "
          f"{len(biasa)} baris, {int(biasa['y'].sum())} positif, prevalensi {biasa['y'].mean() * 100:.2f}%\n" + "=" * 100)

    print("\nTabel 3. S1 2020-2025")
    print(f"  {'model':<22}{'AUC-PR dlm-thn':>15}{'unggul vs klim [95%]':>34}{'AUC-PR gabung':>15}{'ECE':>8}")
    for m in sorted(MODEL, key=lambda m: -dalam_tahun(biasa, m)):
        u = ""
        if m != "klimatologi":
            lo, hi = boot_dalam_tahun(biasa, m, "klimatologi")
            u = f"{dalam_tahun(biasa, m, 'klimatologi'):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        e = f"{ece(biasa['y'].to_numpy(), biasa[m].to_numpy()):.3f}" if m in PELUANG else "—"
        print(f"  {NAMA[m]:<22}{dalam_tahun(biasa, m):>15.3f}{u:>34}{ap(biasa['y'], biasa[m]):>15.3f}{e:>8}")

    print("\nTabel 4. S2 tahun ekstrem disisihkan: keunggulan atas klimatologi dalam tahun itu [95%]")
    for t in ("2015", "2014", "2019"):
        dd = s2[s2["tag"] == t]
        teks = []
        for m in ("xgb", "rf", "mlp"):
            lo, hi = boot_dalam_tahun(dd, m, "klimatologi")
            teks.append(f"{NAMA[m]} {dalam_tahun(dd, m, 'klimatologi'):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {t} (prev {dd['y'].mean() * 100:.1f}%, klim AUC-PR {ap(dd['y'], dd['klimatologi']):.3f}): " + "; ".join(teks))
    y = s2["y"].to_numpy()
    print(f"  ECE tahun ekstrem: " + ", ".join(f"{NAMA[m]} {ece(y, s2[m].to_numpy()):.3f}" for m in ("logistik", "rf", "xgb", "mlp")))

    print("\nTabel 5. S3 transfer antarpulau: rugi gradient boosting, dalam-tahun, baris sama [95%]")
    kunci = ["gid", "tahun", "bulan", "tag"]
    for g in sorted(s3["pulau"].unique()) if len(s3) else []:
        a = s1[s1["pulau"] == g][kunci + ["y", "xgb", "klimatologi"]].copy()
        b = s3[s3["pulau"] == g][["gid", "tahun", "bulan", "xgb"]].rename(columns={"xgb": "xgb_s3"})
        m = a.merge(b, on=["gid", "tahun", "bulan"])
        if m["y"].sum() < 5:
            continue
        lo, hi = boot_dalam_tahun(m, "xgb", "xgb_s3")
        print(f"  {g:<14} positif {int(m['y'].sum()):>4}  S1 {dalam_tahun(m, 'xgb'):.3f}  S3 {dalam_tahun(m, 'xgb_s3'):.3f}"
              f"  rugi {dalam_tahun(m, 'xgb', 'xgb_s3'):+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"  S3 vs klim {dalam_tahun(m, 'xgb_s3', 'klimatologi'):+.3f}")
print("\nselesai.")
