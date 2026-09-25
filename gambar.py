"""
Gambar untuk Artikel 1. Menghasilkan PNG 300 dpi dan PDF di folder Gambar/.

Aturan yang dipatuhi:
  - Tidak ada sumbu ganda. Dua besaran berbeda skala -> dua panel berbagi sumbu-x.
  - Warna kategorikal hanya untuk dua model yang menjadi cerita; lima sisanya kelabu.
    Palet #1F5FA8 dan #C1442A sudah lolos pemeriksaan CVD: dE 19,7 protan, 28,9 normal.
  - Identitas lewat label langsung, bukan warna semata. Aman untuk cetak hitam-putih.
  - Kisi dan sumbu resesif. Garis 1,6 pt. Penanda >= 5 pt.

Pakai:
    python gambar.py DL_FIRE_SV-C2_792597/panel_bulanan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt

JISEBI meminta PNG minimal 330 ppi. Untuk versi itu, arahkan keluaran ke folder lain supaya
Gambar/ yang dipakai naskah EMS tidak tertimpa:
    GAMBAR_DPI=400 GAMBAR_OUT=Kirim_JISEBI/gambar python gambar.py <panel> <oni>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

sys.argv = sys.argv if len(sys.argv) > 2 else sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "percobaan.py")).read().split("if __name__")[0])

BIRU, MERAH = "#1F5FA8", "#C1442A"
KELABU, KELABU_MUDA = "#6B7280", "#B8BCC4"
TINTA, TINTA_2 = "#1F2328", "#5A6169"
SOROT = {"klimatologi": MERAH, "gradient-boosting": BIRU}

# Nama tampilan mengikuti tabel naskah (bahasa Inggris); kunci internal tetap nama di percobaan.py.
LABEL = {"klimatologi": "Climatology", "gradient-boosting": "Gradient boosting",
         "persistence": "Persistence", "seasonal-naive": "Seasonal naive",
         "rasio-analog": "Ratio scaling", "regresi-ONI": "Logistic, ONI",
         "regresi-penuh": "Logistic, full"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": int(os.environ.get("GAMBAR_DPI", 300)), "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": KELABU_MUDA, "axes.linewidth": 0.8, "axes.labelcolor": TINTA,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": "#E7E9EC", "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("GAMBAR_OUT", "Gambar"))
# GAMBAR_SKALA < 1 memperkecil kanvas dengan ukuran huruf tetap, jadi huruf relatif membesar.
# Dipakai untuk JIKI (dua kolom, huruf gambar 6-8 pt): GAMBAR_SKALA=0.8.
SKALA = float(os.environ.get("GAMBAR_SKALA", 1.0))


def uk(w, h):
    return (w * SKALA, h * SKALA)
os.makedirs(OUT, exist_ok=True)


def simpan(fig, nama):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nama}.{ext}"))
    plt.close(fig)
    print(f"  tersimpan: {os.path.relpath(OUT)}/{nama}.png dan .pdf")


def bersih(ax, kisi="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=kisi, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


# ============================================================ muat & hitung

panel_path, oni_path = sys.argv[1], sys.argv[2]
df, ambang = muat(panel_path, oni_path)
print(f"panel: {len(df)} baris, ambang > {ambang:.0f}")

nama_model = list(skor_model(df[df["tahun"] < 2019], df.head(1)).keys())
kum = {n: {"y": [], "raw": [], "cal": []} for n in nama_model}
per_tahun = {n: {} for n in nama_model}

for th in range(2019, 2026):
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    skor = skor_model(latih, uji)
    kal = kalibrator_oof(df, th, nama_model)
    for n, (sl, su) in skor.items():
        iso = kal[n]
        y = uji["y"].to_numpy()
        kum[n]["y"].append(y); kum[n]["raw"].append(su); kum[n]["cal"].append(iso.predict(su))
        if y.sum() > 0:
            ap = average_precision_score(y, su); prev = y.mean()
            per_tahun[n][th] = (ap - prev) / (1 - prev)

for n in nama_model:
    for k in ("y", "raw", "cal"):
        kum[n][k] = np.concatenate(kum[n][k])

rng = np.random.default_rng(42)
Y = kum[nama_model[0]]["y"]
pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
bidx = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])

metrik = pd.DataFrame([{
    "model": n,
    "roc_lazim": roc_auc_score(Y[bidx], kum[n]["raw"][bidx]),
    "pr_op": average_precision_score(Y, kum[n]["raw"]),
    "ece": ece(Y, kum[n]["cal"]),
} for n in nama_model])

# tahun ekstrem
ekstrem = {}
for th in (2014, 2015):
    sisa, uji = df[df["tahun"] != th], df[df["tahun"] == th]
    skor = skor_model(sisa, uji)
    y = uji["y"].to_numpy(); prev = y.mean()
    ekstrem[th] = {n: (average_precision_score(y, skor[n][1]) - prev) / (1 - prev) for n in nama_model}

# ============================================================ Gambar 1

bulanan = df.groupby("bulan", observed=True).agg(tp=("titik_panas", "sum")).reset_index()
bulanan["t"] = pd.PeriodIndex(bulanan["bulan"], freq="M").to_timestamp()
oni_ser = df.groupby("bulan", observed=True)["oni_lag1"].first().reset_index()
oni_ser["t"] = pd.PeriodIndex(oni_ser["bulan"], freq="M").to_timestamp()

fig, (a1, a2) = plt.subplots(2, 1, figsize=uk(7.2, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
a1.fill_between(bulanan["t"], bulanan["tp"], color=MERAH, alpha=0.18, zorder=2)
a1.plot(bulanan["t"], bulanan["tp"], color=MERAH, lw=1.6, zorder=3)
a1.set_ylabel("Hotspots per month")
a1.set_title("(a) VIIRS S-NPP hotspots, 14 districts of West Kalimantan", loc="left")
bersih(a1)
for th, lbl in ((2014, "2014"), (2015, "2015"), (2019, "2019")):
    m = bulanan[bulanan["bulan"].str.startswith(str(th))]
    if len(m):
        i = m["tp"].idxmax()
        a1.annotate(lbl, (bulanan.loc[i, "t"], bulanan.loc[i, "tp"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=TINTA_2, weight="bold")

a2.axhline(0, color=KELABU_MUDA, lw=0.8, zorder=1)
a2.plot(oni_ser["t"], oni_ser["oni_lag1"], color=KELABU, lw=1.6, zorder=3)
a2.fill_between(oni_ser["t"], 0, oni_ser["oni_lag1"],
                where=oni_ser["oni_lag1"] > 0, color=MERAH, alpha=0.15, zorder=2)
a2.fill_between(oni_ser["t"], 0, oni_ser["oni_lag1"],
                where=oni_ser["oni_lag1"] < 0, color=BIRU, alpha=0.15, zorder=2)
a2.set_ylabel("ONI (°C)")
a2.set_title("(b) Oceanic Niño Index", loc="left")
bersih(a2)
simpan(fig, "gambar1_deret_waktu")

# ============================================================ Gambar 2 — kunci
# Protokol lazim dievaluasi ulang 300 kali dengan undian penyeimbangan berbeda,
# karena satu undian tidak menghasilkan peringkat yang dapat diulang.

B = 300
rng2 = np.random.default_rng(7)
roc_boot = {n: [] for n in nama_model}
for _ in range(B):
    idx = np.concatenate([pos, rng2.choice(neg, size=len(pos), replace=False)])
    for n in nama_model:
        roc_boot[n].append(roc_auc_score(Y[idx], kum[n]["raw"][idx]))
roc_boot = {n: np.array(v) for n, v in roc_boot.items()}

peringkat_boot = {n: [] for n in nama_model}
for b in range(B):
    urut = sorted(nama_model, key=lambda n: -roc_boot[n][b])
    for r, n in enumerate(urut, 1):
        peringkat_boot[n].append(r)
peringkat_boot = {n: np.array(v) for n, v in peringkat_boot.items()}

ur_o = metrik.sort_values("pr_op", ascending=False)["model"].tolist()

# Dua panel: (a) sebaran peringkat antar undian, (b) perpindahan peringkat antar protokol.
# Batang rentang tidak dipakai lagi karena saling menutupi pada satu sumbu.

urut_rerata = sorted(nama_model, key=lambda n: peringkat_boot[n].mean())

fig, (pa, pb) = plt.subplots(1, 2, figsize=uk(9.6, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})

# --- (a) sebaran peringkat 300 undian ---
rj = np.random.default_rng(11)
for i, n in enumerate(urut_rerata):
    pbk = peringkat_boot[n]
    c = SOROT.get(n, KELABU_MUDA)
    jit = rj.normal(0, 0.10, size=len(pbk))
    pa.plot(pbk + rj.normal(0, 0.07, size=len(pbk)), np.full(len(pbk), i) + jit,
            "o", ms=2.2, color=c, alpha=0.16 if n in SOROT else 0.10, zorder=2)
    lo, hi = np.percentile(pbk, [5, 95])
    pa.plot([lo, hi], [i, i], color=c, lw=2.0, alpha=0.85, zorder=3, solid_capstyle="round")
    pa.plot([pbk.mean()], [i], "o", color=c, ms=7 if n in SOROT else 5.5, zorder=4,
            markeredgecolor="white", markeredgewidth=1.0)
pa.set_yticks(range(len(urut_rerata)))
pa.set_yticklabels([LABEL[n] for n in urut_rerata], fontsize=8.5)
for lbl, n in zip(pa.get_yticklabels(), urut_rerata):
    if n in SOROT:
        lbl.set_color(SOROT[n]); lbl.set_weight("bold")
pa.set_xlim(0.4, 7.6); pa.set_xticks(range(1, 8))
pa.set_ylim(len(urut_rerata) - 0.5, -0.5)
pa.set_xlabel("Rank under the prevailing protocol")
pa.set_title("(a) Rank across balancing draws", loc="left")
pa.grid(axis="x", alpha=0.9, zorder=0); pa.set_axisbelow(True)
for s in ("top", "right", "left"):
    pa.spines[s].set_visible(False)
pa.tick_params(axis="y", length=0)

# --- (b) perpindahan peringkat antar protokol ---
for n in nama_model:
    y1, y2 = peringkat_boot[n].mean(), ur_o.index(n) + 1
    c = SOROT.get(n, KELABU_MUDA)
    lw = 2.2 if n in SOROT else 1.2
    z = 4 if n in SOROT else 2
    pb.plot([0, 1], [y1, y2], color=c, lw=lw, zorder=z, solid_capstyle="round")
    pb.plot([0, 1], [y1, y2], "o", color=c, ms=6 if n in SOROT else 4.5, zorder=z + 1)

# label kanan saja, agar tidak bertumpuk di kiri
for n in nama_model:
    y2 = ur_o.index(n) + 1
    c = SOROT.get(n, KELABU_MUDA)
    pb.text(1.06, y2, LABEL[n], ha="left", va="center", fontsize=8.5,
            color=c if n in SOROT else TINTA_2,
            weight="bold" if n in SOROT else "normal")

pb.set_xlim(-0.12, 1.62); pb.set_ylim(7.6, 0.4)
pb.set_xticks([0, 1])
pb.set_xticklabels(["prevailing\n(mean of 300 draws)", "operational\n(natural prevalence)"],
                   fontsize=8.5, color=TINTA)
pb.set_yticks(range(1, 8)); pb.set_ylabel("Rank")
pb.set_title("(b) Rank change between protocols", loc="left")
for s in ("top", "right", "bottom", "left"):
    pb.spines[s].set_visible(False)
pb.tick_params(length=0)

# Judul dan keterangan titik tidak digambar di kanvas: pedoman Elsevier (RSASE) meminta judul
# dan penjelasan simbol berada di keterangan gambar, bukan di gambar.
fig.tight_layout()
simpan(fig, "gambar2_peringkat_berbalik")

# ringkasan ketidakstabilan untuk dikutip di artikel
balik_dist = []
for b in range(B):
    ur_l = sorted(nama_model, key=lambda n: -roc_boot[n][b])
    c = 0
    for i in range(len(nama_model)):
        for j in range(i + 1, len(nama_model)):
            a, bb = nama_model[i], nama_model[j]
            if (ur_l.index(a) - ur_l.index(bb)) * (ur_o.index(a) - ur_o.index(bb)) < 0:
                c += 1
    balik_dist.append(c)
balik_dist = np.array(balik_dist)
print(f"  pasangan berbalik atas {B} undian: median {int(np.median(balik_dist))}, "
      f"rentang {balik_dist.min()}-{balik_dist.max()}, "
      f"nol pembalikan pada {(balik_dist == 0).sum()} dari {B} undian")
for n in nama_model:
    pb = peringkat_boot[n]
    print(f"    {n:<20} peringkat rerata {pb.mean():.2f}  rentang {pb.min()}-{pb.max()}")

# ============================================================ Gambar 3

fig, (a1, a2) = plt.subplots(1, 2, figsize=uk(7.4, 3.4))
for n in nama_model:
    c = SOROT.get(n, KELABU_MUDA); lw = 2.0 if n in SOROT else 1.0
    z = 4 if n in SOROT else 2
    fpr, tpr, _ = roc_curve(Y[bidx], kum[n]["raw"][bidx])
    a1.plot(fpr, tpr, color=c, lw=lw, zorder=z)
    pr, rc, _ = precision_recall_curve(Y, kum[n]["raw"])
    a2.plot(rc, pr, color=c, lw=lw, zorder=z)
a1.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=KELABU_MUDA, lw=0.9)
a2.axhline(Y.mean(), ls=(0, (3, 3)), color=KELABU_MUDA, lw=0.9)
a2.text(0.98, Y.mean() + 0.02, f"random baseline {Y.mean():.3f}", ha="right", fontsize=7.5, color=TINTA_2)
a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
a1.set_title("(a) ROC curves, balanced test set", loc="left")
a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
a2.set_title("(b) Precision-recall curves, natural prevalence", loc="left")
for a in (a1, a2):
    bersih(a, kisi="both")
for n, c in SOROT.items():
    a2.plot([], [], color=c, lw=2.0, label=LABEL[n])
a2.plot([], [], color=KELABU_MUDA, lw=1.0, label="five other models")
a2.legend(loc="upper right")
simpan(fig, "gambar3_kurva_roc_pr")

# ============================================================ Gambar 4

fig, ax = plt.subplots(figsize=uk(4.6, 4.2))
ax.plot([0, 1], [0, 1], ls=(0, (3, 3)), color=KELABU_MUDA, lw=0.9, zorder=1)
tepi = np.linspace(0, 1, 11)
for n in nama_model:
    p, y = kum[n]["cal"], Y
    xs, ys = [], []
    for i in range(10):
        m = (p >= tepi[i]) & (p <= tepi[i + 1] if i == 9 else p < tepi[i + 1])
        if m.sum() >= 15:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    c = SOROT.get(n, KELABU_MUDA); lw = 2.0 if n in SOROT else 1.0
    ax.plot(xs, ys, "-o", color=c, lw=lw, ms=5 if n in SOROT else 3.5,
            zorder=4 if n in SOROT else 2)
ax.set_xlabel("Predicted probability (after isotonic calibration)")
ax.set_ylabel("Observed frequency")
bersih(ax, kisi="both")
e = dict(zip(metrik["model"], metrik["ece"]))
ax.text(0.03, 0.95, f"ECE  Climatology {e['klimatologi']:.4f}\n"
                    f"        Gradient boosting {e['gradient-boosting']:.4f}",
        transform=ax.transAxes, va="top", fontsize=8, color=TINTA_2)
simpan(fig, "gambar4_reliabilitas")

# ============================================================ Gambar 5

tahun = sorted(per_tahun["klimatologi"].keys())
fig, ax = plt.subplots(figsize=uk(7.0, 3.6))
x = np.arange(len(tahun) + 2)
lbl = [str(t) for t in tahun] + ["2014*", "2015*"]
for n, c in SOROT.items():
    v = [per_tahun[n][t] for t in tahun] + [ekstrem[2014][n], ekstrem[2015][n]]
    ax.plot(x, v, "-o", color=c, lw=1.8, ms=6, label=LABEL[n], zorder=4)
for n in nama_model:
    if n in SOROT:
        continue
    v = [per_tahun[n][t] for t in tahun] + [ekstrem[2014][n], ekstrem[2015][n]]
    ax.plot(x, v, color=KELABU_MUDA, lw=0.9, zorder=2)
ax.axvspan(len(tahun) - 0.5, len(tahun) + 1.5, color="#F3F4F6", zorder=0)
ax.text(len(tahun) + 0.5, ax.get_ylim()[1], "extreme years\nwithheld from training",
        ha="center", va="top", fontsize=8, color=TINTA_2)
ax.set_xticks(x); ax.set_xticklabels(lbl)
ax.set_ylabel("Normalised AUC-PR skill")
ax.set_xlabel("Test year")
bersih(ax)
ax.plot([], [], color=KELABU_MUDA, lw=0.9, label="five other models")
ax.legend(loc="lower left", ncol=3)
simpan(fig, "gambar5_kinerja_per_tahun")

print("\nselesai. lima gambar di folder Gambar/")
