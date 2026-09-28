"""
Empat gambar artikel Kalimantan, dari DL_FIRE_NASIONAL/prediksi_kalimantan.csv dan panel Kalimantan.

  gambar_kal1_peta          : 55 kabupaten, jumlah deteksi 2013-2025 (skala log)
  gambar_kal2_peringkat     : peringkat AUC-PR tujuh model per provinsi, dua sasaran
  gambar_kal3_transfer      : AUC-PR gradient boosting ID / LOPO / LOKAL per provinsi, dua sasaran
  gambar_kal4_ekstrem       : skill GB - klimatologi pada 2014 dan 2015 per provinsi, dua sasaran

Gaya mengikuti gambar.py: dua warna sorotan (biru GB, merah klimatologi) dan kelabu, tanpa
sumbu ganda, label langsung. Judul gambar TIDAK digambar di kanvas (pedoman Elsevier/Springer:
judul di keterangan). Kalimantan Utara diberi tanda karena positifnya sedikit.

Pakai:
    python gambar_kalimantan.py DL_FIRE_NASIONAL/prediksi_kalimantan.csv \
        DL_FIRE_NASIONAL/panel_kalimantan.csv DL_FIRE_SV-C2_792597/gadm41_IDN_2.json

Keluaran: Gambar_Kalimantan/*.png (300 dpi) dan *.pdf
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Polygon
from sklearn.metrics import average_precision_score as ap

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
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, os.environ.get("GAMBAR_OUT", "Gambar_Kalimantan"))
os.makedirs(OUT, exist_ok=True)

PROV = ["KalimantanBarat", "KalimantanTengah", "KalimantanSelatan", "KalimantanTimur", "KalimantanUtara"]
NAMA_PROV = {"KalimantanBarat": "West", "KalimantanTengah": "Central", "KalimantanSelatan": "South",
             "KalimantanTimur": "East", "KalimantanUtara": "North†"}
MODEL = ["gradient-boosting", "regresi-penuh", "klimatologi", "persistence", "rasio-analog",
         "regresi-ONI", "seasonal-naive"]
LABEL = {"klimatologi": "Climatology", "gradient-boosting": "Gradient boosting",
         "persistence": "Persistence", "seasonal-naive": "Seasonal naive",
         "rasio-analog": "Ratio scaling", "regresi-ONI": "Logistic, ONI",
         "regresi-penuh": "Logistic, full"}
SASARAN = [("gabungan", "(a) Pooled threshold"), ("kabupaten", "(b) Per-district threshold")]

pred_path, panel_path, gadm_path = sys.argv[1:4]
P = pd.read_csv(pred_path)
R = P[P["skema"] == "rolling"]


def simpan(fig, nama):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nama}.{ext}"))
    plt.close(fig)
    print(f"  tersimpan: {nama}.png dan .pdf")


def boot_ap(y, s, n=1000, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            out.append(ap(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


# ============================================================ Gambar 1: peta
panel = pd.read_csv(panel_path)
panel = panel[(panel["tahun"] >= 2013) & (panel["tahun"] <= 2025)]
total = panel.groupby("kabupaten")["titik_panas"].sum()
gj = json.load(open(gadm_path, encoding="utf-8"))
fitur_kal = [f for f in gj["features"] if f["properties"]["NAME_1"].startswith("Kalimantan")]
assert len(fitur_kal) == 55, len(fitur_kal)

patches, nilai, per_prov_xy = [], [], {}
for f in fitur_kal:
    nama = f["properties"]["NAME_2"]
    geom = f["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    v = total.get(nama, np.nan)
    for poly in polys:
        luar = np.array(poly[0])
        patches.append(Polygon(luar, closed=True))
        nilai.append(v)
        per_prov_xy.setdefault(f["properties"]["NAME_1"], []).append(luar)
nilai = np.array(nilai, dtype=float)
assert not np.isnan(nilai).any(), "nama kabupaten GADM tidak cocok dengan panel"

peta_warna = LinearSegmentedColormap.from_list("api", ["#FBEFEA", "#E8A68F", MERAH, "#6E1F12"])
fig, ax = plt.subplots(figsize=(6.4, 5.6))
pc = PatchCollection(patches, cmap=peta_warna, norm=LogNorm(vmin=max(nilai.min(), 1), vmax=nilai.max()),
                     edgecolor="white", linewidth=0.4)
pc.set_array(nilai)
ax.add_collection(pc)
for prov, arr in per_prov_xy.items():
    xy = np.vstack(arr)
    ax.text(np.median(xy[:, 0]), np.median(xy[:, 1]), NAMA_PROV[prov].replace("†", ""),
            ha="center", va="center", fontsize=9, weight="bold", color=TINTA,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
ax.autoscale_view(); ax.set_aspect("equal")
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("VIIRS detections, 2013–2025 (log scale)")
simpan(fig, "gambar_kal1_peta")

# ============================================================ Gambar 2: peringkat
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
peta_rank = LinearSegmentedColormap.from_list("rank", [BIRU, "#9DB8DA", "#F3F4F6"])
for ax, (s, judul) in zip(axes, SASARAN):
    d = R[R["sasaran"] == s]
    kolom = ["Kalimantan"] + PROV
    M = np.zeros((len(MODEL), len(kolom)))
    for j, k in enumerate(kolom):
        dd = d if k == "Kalimantan" else d[d["provinsi"] == k]
        v = {m: ap(dd["y"], dd[m]) for m in MODEL}
        urut = sorted(v, key=lambda m: -v[m])
        for i, m in enumerate(MODEL):
            M[i, j] = urut.index(m) + 1
    ax.imshow(M, cmap=peta_rank, vmin=1, vmax=7, aspect="auto")
    for i in range(len(MODEL)):
        for j in range(len(kolom)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] <= 2 else TINTA, weight="bold" if M[i, j] == 1 else "normal")
    ax.set_xticks(range(len(kolom)))
    ax.set_xticklabels(["All"] + [NAMA_PROV[p] for p in PROV], fontsize=8.5)
    ax.axvline(0.5, color="white", lw=3)
    ax.set_title(judul, loc="left")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_yticks(range(len(MODEL)))
axes[0].set_yticklabels([LABEL[m] for m in MODEL], fontsize=8.5)
fig.tight_layout()
simpan(fig, "gambar_kal2_peringkat")

# ============================================================ Gambar 3: transfer
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
CARA = [("gradient-boosting", "Trained on all Kalimantan", BIRU, "o"),
        ("gb_lopo", "Trained without the province", "#7FA3D1", "s"),
        ("gb_lokal", "Trained on the province only", TINTA_2, "D"),
        ("klimatologi", "Climatology (local)", MERAH, "^")]
for ax, (s, judul) in zip(axes, SASARAN):
    d = R[R["sasaran"] == s]
    for k, (kol, lab, warna, penanda) in enumerate(CARA):
        for j, p in enumerate(PROV):
            dd = d[(d["provinsi"] == p) & d[kol].notna()]
            y, sc = dd["y"].to_numpy(), dd[kol].to_numpy()
            x = j + (k - 1.5) * 0.17
            v = ap(y, sc); lo, hi = boot_ap(y, sc)
            ax.plot([x, x], [lo, hi], color=warna, lw=1.2, alpha=0.8)
            ax.plot(x, v, penanda, color=warna, ms=5.5, label=lab if j == 0 else None,
                    markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(range(len(PROV)))
    ax.set_xticklabels([NAMA_PROV[p] for p in PROV])
    ax.set_title(judul, loc="left")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("AUC-PR (95% bootstrap interval)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.07, 1, 1])
simpan(fig, "gambar_kal3_transfer")

# ============================================================ Gambar 4: tahun ekstrem


def skill(y, s):
    p = y.mean(); return (ap(y, s) - p) / (1 - p)


fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
for ax, (s, judul) in zip(axes, SASARAN):
    for k, (th, warna, penanda) in enumerate(((2014, TINTA_2, "o"), (2015, MERAH, "s"))):
        d = P[(P["sasaran"] == s) & (P["skema"] == f"ekstrem_{th}")]
        for j, p in enumerate(PROV):
            dd = d[d["provinsi"] == p]
            y, g, c = dd["y"].to_numpy(), dd["gradient-boosting"].to_numpy(), dd["klimatologi"].to_numpy()
            if y.sum() < 3 or y.sum() == len(y):
                continue
            rng = np.random.default_rng(7); bb = []
            for _ in range(1000):
                i = rng.integers(0, len(y), len(y))
                if 0 < y[i].sum() < len(i):
                    bb.append(skill(y[i], g[i]) - skill(y[i], c[i]))
            lo, hi = np.percentile(bb, [2.5, 97.5])
            x = j + (k - 0.5) * 0.25
            ax.plot([x, x], [lo, hi], color=warna, lw=1.2)
            ax.plot(x, skill(y, g) - skill(y, c), penanda, color=warna, ms=6,
                    label=str(th) if j == 0 else None, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=KELABU, lw=0.9)
    ax.set_xticks(range(len(PROV)))
    ax.set_xticklabels([NAMA_PROV[p] for p in PROV])
    ax.set_title(judul, loc="left")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Skill difference, gradient boosting − climatology")
axes[0].legend(loc="upper left", title="Withheld year", title_fontsize=7.5)
fig.tight_layout()
simpan(fig, "gambar_kal4_ekstrem")
print("selesai.")
