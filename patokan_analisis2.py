"""
Artikel 2 (PeatFireBench) — analisis lanjutan dari DL_FIRE_NASIONAL/patokan_prediksi.csv
(Hasil_Patokan_1 bagian 5).

  1. Keunggulan gradient boosting atas klimatologi per tahun uji: S1 tiap tahun 2019-2025,
     gabungan tahun non-ekstrem (2020-2025), dan tiga tahun ekstrem disisihkan (S2). Pembanding
     pada baris yang sama, jadi kebal terhadap perbedaan prevalensi antartahun.
  2. Skor probabilistik model yang dipelajari (logistik, rf, xgb, mlp): Brier, Brier skill
     terhadap prevalensi masa latih (konstanta), dan ECE 10 bin. Baseline lokal tidak
     dinilai di sini karena keluarannya hitungan titik panas, bukan peluang.
  3. Dua gambar ke Gambar_Patokan/: keunggulan per tahun lawan jumlah deteksi nasional, dan
     diagram reliabilitas gradient boosting (S1 non-ekstrem lawan tahun ekstrem).

Pakai:  python patokan_analisis2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as ap

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
OUT = os.path.join(BASE, "Gambar_Patokan")
os.makedirs(OUT, exist_ok=True)
P = pd.read_csv(os.path.join(D, "patokan_prediksi.csv"), dtype={"tag": str})
F = pd.read_csv(os.path.join(D, "panel_nasional_fitur_bersih.csv"),
                usecols=["gid", "tahun", "y_kabupaten", "y_gabungan", "ada_positif"])
deteksi = pd.read_csv(os.path.join(D, "panel_nasional_bersih.csv"),
                      usecols=["tahun", "titik_panas"]).groupby("tahun")["titik_panas"].sum()
EKSTREM = ("2015", "2014", "2019")
PELUANG = ["logistik", "rf", "xgb", "mlp"]
NAMA = {"logistik": "Logistic (ONI, DMI)", "rf": "Random forest", "xgb": "Gradient boosting", "mlp": "MLP"}

BIRU, MERAH = "#1F5FA8", "#C1442A"
KELABU, KELABU_MUDA = "#6B7280", "#B8BCC4"
TINTA, TINTA_2 = "#1F2328", "#5A6169"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": int(os.environ.get("GAMBAR_DPI", 300)), "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": KELABU_MUDA, "axes.linewidth": 0.8, "axes.labelcolor": TINTA,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": "#E7E9EC", "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 8,
})
SASARAN = [("y_gabungan", "(a) Pooled threshold"), ("y_kabupaten", "(b) Per-district threshold")]


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


from functools import lru_cache


@lru_cache(maxsize=None)
def prevalensi_latih(sas, skema, tag):
    f = F[F["ada_positif"] == 1] if sas == "y_kabupaten" else F
    th = int(tag)
    latih = f[f["tahun"] < th] if skema == "S1" else f[f["tahun"] != th]
    return latih[sas].mean()


unggul = {}
for sas, _ in SASARAN:
    d0 = P[(P["sasaran"] == sas) & P["skema"].isin(["S1", "S2"])]
    if sas == "y_kabupaten":
        d0 = d0[d0["ada_positif"] == 1]
    print("\n" + "=" * 90 + f"\nSASARAN {sas}\n" + "=" * 90)

    # 1. keunggulan atas klimatologi per tahun
    print("\n1. Keunggulan AUC-PR gradient boosting atas klimatologi, baris sama [95% klaster]")
    print(f"  {'uji':<22}{'deteksi':>10}{'prev':>7}{'klim':>7}{'xgb':>7}   selisih")
    kelompok = [(f"S1 {t}", d0[(d0["skema"] == "S1") & (d0["tag"] == str(t))]) for t in range(2019, 2026)]
    kelompok.append(("S1 non-ekstrem 20-25", d0[(d0["skema"] == "S1") & (d0["tahun"] >= 2020)]))
    kelompok += [(f"S2 {t} disisihkan", d0[(d0["skema"] == "S2") & (d0["tag"] == t)]) for t in EKSTREM]
    baris = []
    for nama, dd in kelompok:
        a, k = ap(dd["y"], dd["xgb"]), ap(dd["y"], dd["klimatologi"])
        lo, hi = boot_beda(dd, "xgb", "klimatologi")
        th = int(nama.split()[1]) if nama.split()[1].isdigit() else None
        det = deteksi.get(th, np.nan) if th else np.nan
        print(f"  {nama:<22}{det:>10.0f}{dd['y'].mean() * 100:>6.1f}%{k:>7.3f}{a:>7.3f}   "
              f"{a - k:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        baris.append(dict(nama=nama, tahun=th, deteksi=det, beda=a - k, lo=lo, hi=hi))
    unggul[sas] = pd.DataFrame(baris)

    # 2. skor probabilistik
    print("\n2. Skor probabilistik (Brier, BSS terhadap prevalensi masa latih, ECE 10 bin)")
    print(f"  {'model':<22}" + "".join(f"{h:>24}" for h in ("S1 non-ekstrem 20-25", "S2 ekstrem (gabungan)")))
    blok = {"S1 non-ekstrem 20-25": d0[(d0["skema"] == "S1") & (d0["tahun"] >= 2020)],
            "S2 ekstrem (gabungan)": d0[d0["skema"] == "S2"]}
    for m in PELUANG:
        teks = ""
        for nb_, dd in blok.items():
            y, p = dd["y"].to_numpy(), dd[m].to_numpy()
            ref = np.array([prevalensi_latih(sas, s, t) for s, t in zip(dd["skema"], dd["tag"])])
            bs, bs_ref = np.mean((p - y) ** 2), np.mean((ref - y) ** 2)
            teks += f"   {bs:.4f} {1 - bs / bs_ref:+.3f} {ece(y, p):.3f}"
        print(f"  {NAMA[m]:<22}{teks}")
    print("  (kolom: Brier, BSS, ECE)")

    # 3b. diagram reliabilitas gradient boosting
    unggul[sas + "_rel"] = {k: (v["y"].to_numpy(), v["xgb"].to_numpy()) for k, v in blok.items()}


def simpan(fig, nama):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nama}.{ext}"))
    plt.close(fig)
    print(f"  tersimpan: {nama}.png dan .pdf")


print("\nGambar:")
# geser label tahun yang bertumpuk di sekitar 45-85 ribu deteksi (titik data tidak digeser)
GESER = {("y_gabungan", 2021, False): (-24, 2), ("y_gabungan", 2020, False): (5, 6),
         ("y_gabungan", 2025, False): (5, -8), ("y_kabupaten", 2021, False): (6, -9),
         ("y_kabupaten", 2022, False): (-26, 2), ("y_kabupaten", 2025, False): (6, 4),
         ("y_kabupaten", 2024, False): (6, -7)}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, (sas, judul) in zip(axes, SASARAN):
    u = unggul[sas]
    for _, r in u[u["tahun"].notna()].iterrows():
        eks = r["nama"].startswith("S2")
        c = MERAH if eks else BIRU
        ax.errorbar(r["deteksi"] / 1e3, r["beda"], yerr=[[r["beda"] - r["lo"]], [r["hi"] - r["beda"]]],
                    fmt="o" if not eks else "s", color=c, ms=4.5, elinewidth=1, capsize=0)
        dx, dy = GESER.get((sas, int(r["tahun"]), eks), (4, 3))
        ax.annotate(str(int(r["tahun"])), (r["deteksi"] / 1e3, r["beda"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, color=c)
    ax.axhline(0, color=KELABU, lw=0.8)
    ax.set_title(judul, loc="left")
    ax.grid(True, axis="y")
fig.supxlabel("National VIIRS detections in test year (thousands)", fontsize=9, y=-0.02)
axes[0].set_ylabel("AUC-PR, gradient boosting − climatology")
axes[1].plot([], [], "o", color=BIRU, ms=4.5, label="Chronological split")
axes[1].plot([], [], "s", color=MERAH, ms=4.5, label="Extreme year withheld")
axes[1].legend(loc="upper right")
simpan(fig, "gambar_pat2_ekstrem")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, (sas, judul) in zip(axes, SASARAN):
    for (nb_, (y, p)), c in zip(unggul[sas + "_rel"].items(), (BIRU, MERAH)):
        b = np.minimum((p * 10).astype(int), 9)
        xs = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 20]
        ys = [y[b == k].mean() for k in range(10) if (b == k).sum() >= 20]
        lab = "Non-extreme years 2020–2025" if nb_.startswith("S1") else "Extreme years withheld"
        ax.plot(xs, ys, "o-", color=c, ms=3.5, lw=1.2, label=lab)
    ax.plot([0, 1], [0, 1], color=KELABU_MUDA, lw=0.8, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(judul, loc="left"); ax.set_xlabel("Predicted probability"); ax.grid(True)
axes[0].set_ylabel("Observed frequency")
axes[1].legend(loc="lower right")
simpan(fig, "gambar_pat3_reliabilitas")
print("\nselesai.")
