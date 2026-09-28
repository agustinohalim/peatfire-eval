"""
Percobaan Kalimantan 1 — apakah temuan Artikel 1 bertahan di lima provinsi Kalimantan?

Artikel lanjutan dari Artikel 1 (Kalimantan Barat saja). Pertanyaannya bukan mengulang
Artikel 1 pada wilayah lebih luas, melainkan: temuan mana yang umum dan mana yang milik satu
provinsi, dan apakah peringkat model berpindah antarprovinsi. Artikel 2 (tolok ukur nasional)
tidak disentuh; panel Kalimantan di sini bukan kontribusi dataset.

  A. Rancangan 2x2 Artikel 1 pada 55 kabupaten, ambang gabungan dan ambang per kabupaten:
     peringkat empat sel, tau, pembalikan atas 300 undian, selisih GB - klimatologi dengan
     selang bootstrap, dan kalibrasi out-of-fold.
  B. Per provinsi: model dilatih pada seluruh Kalimantan, dinilai terpisah di tiap provinsi.
     Apakah peringkat AUC-PR dan pemenang GB lawan klimatologi sama di kelimanya?

Pakai:
    python percobaan_kalimantan.py DL_FIRE_NASIONAL/panel_nasional.csv DL_FIRE_SV-C2_792597/oni.ascii.txt

Panel Kalimantan ditulis ke DL_FIRE_NASIONAL/panel_kalimantan.csv (tidak masuk repo).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py"), encoding="utf-8").read().split("if __name__")[0])

nasional, oni_path = sys.argv[1], sys.argv[2]
p = pd.read_csv(nasional)
kal = p[p["provinsi"].str.startswith("Kalimantan") & (p["tahun"] <= 2025)].copy()
jalur_kal = os.path.join(os.path.dirname(nasional), "panel_kalimantan.csv")
kal.to_csv(jalur_kal, index=False)

df, ambang = muat(jalur_kal, oni_path)
TAHUN_UJI = list(range(2019, 2026))
GB, KL = "gradient-boosting", "klimatologi"
LABEL = {"klimatologi": "Climatology", "gradient-boosting": "Gradient boosting",
         "persistence": "Persistence", "seasonal-naive": "Seasonal naive",
         "rasio-analog": "Ratio scaling", "regresi-ONI": "Logistic, ONI",
         "regresi-penuh": "Logistic, full"}

print("=" * 78)
print(f"PANEL KALIMANTAN: {df['kabupaten'].nunique()} kabupaten, {len(df)} baris "
      f"({df['bulan'].min()} s/d {df['bulan'].max()}), ambang gabungan p90 > {ambang:.0f}")
print("=" * 78)
per_prov = df.groupby("provinsi").agg(kab=("kabupaten", "nunique"), titik=("titik_panas", "sum"),
                                      positif=("y", "mean"))
print(per_prov.to_string(formatters={"positif": "{:.1%}".format}))
jarang = df.groupby("kabupaten")["y"].mean()
print(f"kabupaten yang tak pernah melampaui ambang gabungan: {(jarang == 0).sum()} dari {len(jarang)}; "
      f"yang melampauinya > 33% bulan: {(jarang > 0.33).sum()}")


def lipat(d, kalibrasi=True):
    ys, meta, sk, cal = [], [], None, None
    for th in TAHUN_UJI:
        latih, uji = d[d["tahun"] < th], d[d["tahun"] == th]
        s = skor_model(latih, uji)
        if sk is None:
            sk = {n: [] for n in s}; cal = {n: [] for n in s}
        kal_ = kalibrator_oof(d, th, list(s)) if kalibrasi else None
        for n, (_, su) in s.items():
            sk[n].append(su)
            if kalibrasi:
                cal[n].append(kal_[n].predict(su))
        ys.append(uji["y"].to_numpy())
        meta.append(uji[["provinsi", "kabupaten", "tahun"]])
    return (np.concatenate(ys), pd.concat(meta, ignore_index=True),
            {n: np.concatenate(v) for n, v in sk.items()},
            {n: np.concatenate(v) for n, v in cal.items()} if kalibrasi else None)


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


def boot_beda(y, a, b, n=2000, seed=2026):
    rb = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rb.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        out.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    return np.percentile(out, [2.5, 97.5])


def empat_sel(y, sk, cal, judul):
    nama = list(sk)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(7)
    D = {n: average_precision_score(y, sk[n]) for n in nama}
    C = {n: roc_auc_score(y, sk[n]) for n in nama}
    rD, rC = peringkat(D), peringkat(C)
    A_sum = {n: 0.0 for n in nama}
    rA_list, balik_list, gb_atas = [], [], 0
    for _ in range(300):
        i = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        A = {n: roc_auc_score(y[i], sk[n][i]) for n in nama}
        for n in nama:
            A_sum[n] += A[n] / 300
        rA = peringkat(A)
        rA_list.append(rA); balik_list.append(balik(rA, rD)); gb_atas += A[GB] > A[KL]
    rA_rerata = peringkat({n: -np.mean([r[n] for r in rA_list]) for n in nama})
    print(f"\n{judul}")
    print(f"  n uji {len(y)}, positif {int(y.sum())}, prevalensi {y.mean() * 100:.1f}%")
    print(f"  {'model':<20}{'ROC seimb.':>11}{'ROC asli':>10}{'AUC-PR':>8}{'Brier':>9}{'ECE':>8}   rank A C D")
    for n in sorted(nama, key=lambda n: rD[n]):
        br = brier_score_loss(y, cal[n]) if cal else float("nan")
        ec = ece(y, cal[n]) if cal else float("nan")
        print(f"  {LABEL[n]:<20}{A_sum[n]:>11.3f}{C[n]:>10.3f}{D[n]:>8.3f}{br:>9.4f}{ec:>8.4f}"
              f"      {rA_rerata[n]} {rC[n]} {rD[n]}")
    b = np.array(balik_list)
    print(f"  tau C vs D = {tau(rC, rD):+.3f} ({balik(rC, rD)} pasangan sumbang); pembalikan A vs D: "
          f"median {int(np.median(b))}, rentang {b.min()}-{b.max()}, nol pada {(b == 0).sum()}/300")
    lo, hi = boot_beda(y, sk[GB], sk[KL])
    print(f"  selisih AUC-PR GB - klimatologi {D[GB] - D[KL]:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
          f"GB di atas klimatologi (ROC seimbang) {gb_atas}/300 undian")
    return rD, D


# ============================================================ A. gabungan
y, meta, sk, cal = lipat(df)
rD_gab, D_gab = empat_sel(y, sk, cal, "A1. KALIMANTAN, AMBANG GABUNGAN (p90)")

mentah = pd.read_csv(jalur_kal)
q_kab = mentah.groupby("kabupaten")["titik_panas"].quantile(0.90)
d2 = df.copy()
d2["y"] = (d2["titik_panas"] > d2["kabupaten"].map(q_kab)).astype(int)
y2, meta2, sk2, cal2 = lipat(d2)
rD_kab, D_kab = empat_sel(y2, sk2, cal2, "A2. KALIMANTAN, AMBANG PER KABUPATEN (p90)")
print(f"\n  tau peringkat AUC-PR gabungan vs per kabupaten = {tau(rD_gab, rD_kab):+.3f}")

# ============================================================ B. per provinsi


def per_provinsi(y, meta, sk, judul, rD_acuan):
    print(f"\n{judul}")
    print(f"  {'provinsi':<20}{'pos':>5}{'prev':>7}  {'GB':>6}{'klim':>6}  selisih [95%]           "
          f"tau vs Kalimantan  juara")
    for pv in sorted(meta["provinsi"].unique()):
        m = (meta["provinsi"] == pv).to_numpy()
        if y[m].sum() < 5:
            print(f"  {pv:<20}{int(y[m].sum()):>5}  terlalu sedikit positif"); continue
        D = {n: average_precision_score(y[m], sk[n][m]) for n in sk}
        rD = peringkat(D)
        lo, hi = boot_beda(y[m], sk[GB][m], sk[KL][m])
        juara = min(rD, key=rD.get)
        print(f"  {pv:<20}{int(y[m].sum()):>5}{y[m].mean() * 100:>6.1f}%  {D[GB]:>6.3f}{D[KL]:>6.3f}  "
              f"{D[GB] - D[KL]:+.3f} [{lo:+.3f}, {hi:+.3f}]  {tau(rD, rD_acuan):+.3f}           {LABEL[juara]}")


per_provinsi(y, meta, sk, "B1. PER PROVINSI, AMBANG GABUNGAN (model dilatih pada seluruh Kalimantan)", rD_gab)
per_provinsi(y2, meta2, sk2, "B2. PER PROVINSI, AMBANG PER KABUPATEN", rD_kab)

print("\nselesai.")
