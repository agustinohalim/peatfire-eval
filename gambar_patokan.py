"""
Gambar naskah Artikel 2 (PeatFireBench), dari panel dan putaran panel lengkap.

Sejak 7 Nov 2026 menulis nama gambar_rev* dengan nomor naskah revisi (Naskah_Artikel2_Revisi.md).
Berkas gambar_pat* yang sudah ada adalah gambar naskah terkirim ke Natural Hazards dan tidak ditimpa.

  gambar_rev1_deret        : Fig. 1, deteksi bulanan nasional 2012-2025, tahun uji ekstrem ditandai
  gambar_rev2_peta         : Fig. 2, 498 kabupaten/kota, (a) deteksi VIIRS 2012-2025 (skala log),
                             (b) ambang per kabupaten maks(p90, 10); kabupaten tanpa positif kelabu
  gambar_rev3_ablasi       : Fig. 3, sumbangan tiap kelompok fitur, dalam-tahun lawan digabung lintas
                             tahun, dua sasaran (dari patokan_ablasi.csv)
  gambar_rev4_ekstrem      : Fig. 4, keunggulan dalam-tahun gradient boosting per tahun uji lawan jumlah
                             deteksi nasional, atas MC (simbol kosong) dan atas rujukan pasangan Tabel 2
                             (simbol isi; EF-1m per kabupaten, EF-3m nasional), dua sasaran
  gambar_rev5_reliabilitas : Fig. 5, diagram reliabilitas gradient boosting, tahun biasa 2020-2025 lawan
                             tahun ekstrem disisihkan, dua sasaran

Prediksi dari patokan_revisi_prediksi.csv (klimatologi dan xgb sama dengan berkas lama; berkas ini
menambah klim_frek dan klim_frek1).

Gaya mengikuti gambar_kalimantan.py; judul tidak digambar di kanvas (judul di keterangan).
Pakai:    python gambar_patokan.py
Keluaran: Gambar_Patokan/*.png (300 dpi) dan *.pdf
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Patch, Polygon
from sklearn.metrics import average_precision_score as ap
from kelompok_fitur import ABLASI, KELOMPOK

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
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
OUT = os.path.join(BASE, "Gambar_Patokan")
os.makedirs(OUT, exist_ok=True)
SASARAN = [("y_gabungan", "(a) National threshold"), ("y_kabupaten", "(b) Per-district threshold")]
EKSTREM = ("2015", "2014", "2019")


def simpan(fig, nama):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nama}.{ext}"))
    plt.close(fig)
    print(f"  tersimpan: {nama}.png dan .pdf")


def pilih(P, sas):
    d = P[P["sasaran"] == sas]
    return d[d["ada_positif"] == 1] if sas == "y_kabupaten" else d


def dalam_tahun(d, a, b):
    return float(np.mean([ap(g["y"], g[a]) - ap(g["y"], g[b]) for _, g in d.groupby("tag") if g["y"].sum()]))


def boot_dalam_tahun(d, a, b, n=1000, seed=2026):
    uk = d["gid"].unique(); rng = np.random.default_rng(seed)
    per = {t: g for t, g in d.groupby("tag")}
    idx = {t: {k: np.where(g["gid"].to_numpy() == k)[0] for k in uk} for t, g in per.items()}
    out = []
    for _ in range(n):
        pl = rng.choice(uk, size=len(uk), replace=True); v = []
        for t, g in per.items():
            j = np.concatenate([idx[t][k] for k in pl]); y = g["y"].to_numpy()[j]
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


panel = pd.read_csv(os.path.join(D, "panel_nasional_bersih.csv"))
deteksi = panel.groupby("tahun")["titik_panas"].sum()
P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str},
                usecols=["gid", "tahun", "ada_positif", "y", "sasaran", "skema", "tag", "klimatologi",
                         "klim_frek", "klim_frek1", "xgb"])
P = P[P["skema"] != "Q"]
A = pd.read_csv(os.path.join(D, ABLASI), dtype={"tag": str})
print("Gambar:")

# ============================================================ 1. peta
per_gid = panel.groupby("gid").agg(total=("titik_panas", "sum"), ambang=("ambang_kabupaten", "first"),
                                   ada=("ada_positif", "first"))
gj = json.load(open(os.path.join(BASE, "DL_FIRE_SV-C2_792597", "gadm41_IDN_2.json"), encoding="utf-8"))
fitur = [f for f in gj["features"] if f["properties"]["GID_2"] in per_gid.index]
assert len(fitur) == 498, len(fitur)
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2))
panel_peta = [("total", "Detections 2012–2025", ["#FBEFEA", "#E8A68F", MERAH, "#6E1F12"]),
              ("ambang", "Threshold, detections/month", ["#EAF1FA", "#8DB3E2", BIRU, "#123A68"])]
for ax, (kol, label, warna), huruf in zip(axes, panel_peta, "ab"):
    patches, nilai, abu = [], [], []
    for f in fitur:
        gid = f["properties"]["GID_2"]; g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        v = float(per_gid.loc[gid, kol])
        tanpa = kol == "ambang" and per_gid.loc[gid, "ada"] == 0
        for poly in polys:
            (abu if tanpa else patches).append(Polygon(np.array(poly[0]), closed=True))
            if not tanpa:
                nilai.append(max(v, 1))
    cmap = LinearSegmentedColormap.from_list(kol, warna)
    pc = PatchCollection(patches, cmap=cmap, norm=LogNorm(vmin=max(min(nilai), 1), vmax=max(nilai)),
                         edgecolor="white", linewidth=0.15)
    pc.set_array(np.array(nilai)); ax.add_collection(pc)
    if abu:
        ax.add_collection(PatchCollection(abu, facecolor=KELABU_MUDA, edgecolor="white", linewidth=0.15))
    ax.autoscale_view(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"({huruf})", loc="left")
    cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.01); cb.set_label(label, fontsize=8)
    if kol == "ambang":
        ax.text(0.01, 0.02, "grey: no month above threshold (excluded from per-district metrics)",
                transform=ax.transAxes, fontsize=7, color=TINTA_2)
simpan(fig, "gambar_rev2_peta")

# ============================================================ 2. keunggulan per tahun (Fig. 4)
RUJUKAN = {"y_kabupaten": "klim_frek1", "y_gabungan": "klim_frek"}
GESER = {("y_gabungan", "S1", 2022): (-24, 3), ("y_gabungan", "S1", 2021): (-24, -8),
         ("y_gabungan", "S1", 2020): (5, 3), ("y_gabungan", "S1", 2025): (5, -6),
         ("y_gabungan", "S2", 2019): (-26, -11), ("y_kabupaten", "S1", 2021): (-24, -8),
         ("y_kabupaten", "S1", 2020): (5, 2), ("y_kabupaten", "S1", 2025): (5, -5),
         ("y_kabupaten", "S2", 2019): (-26, -11)}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, (sas, judul) in zip(axes, SASARAN):
    d0 = pilih(P, sas)
    titik = [(t, "S1", d0[(d0["skema"] == "S1") & (d0["tag"] == str(t))]) for t in range(2020, 2026)]
    titik += [(int(t), "S2", d0[(d0["skema"] == "S2") & (d0["tag"] == t)]) for t in EKSTREM]
    geser = 0.012 * deteksi.max() / 1e3
    for th, sk, dd in titik:
        c, mk = (MERAH, "s") if sk == "S2" else (BIRU, "o")
        x = deteksi[th] / 1e3
        for rj, dx, isi in (("klimatologi", -geser, "none"), (RUJUKAN[sas], geser, c)):
            b = ap(dd["y"], dd["xgb"]) - ap(dd["y"], dd[rj])
            lo, hi = boot_dalam_tahun(dd, "xgb", rj)
            ax.errorbar(x + dx, b, yerr=[[b - lo], [hi - b]], fmt=mk, color=c, mfc=isi, ms=4.5,
                        elinewidth=1, capsize=0)
            print(f"    {sas} {sk} {th} {rj}: {b:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if rj != "klimatologi":
                ax.annotate(str(th), (x + dx, b), xytext=GESER.get((sas, sk, th), (4, -3)), textcoords="offset points", fontsize=7,
                            color=c)
    ax.axhline(0, color=KELABU, lw=0.8); ax.set_title(judul, loc="left"); ax.grid(True, axis="y")
    ax.set_xlim(-70, 870)
axes[0].set_ylabel("AUC-PR, gradient boosting − reference")
fig.supxlabel("National VIIRS detections in test year (thousands)", fontsize=9, y=-0.02)
fig.legend(handles=[
    plt.Line2D([], [], ls="", marker="o", color=KELABU, mfc="none", ms=4.5, label="Over MC"),
    plt.Line2D([], [], ls="", marker="o", color=KELABU, ms=4.5, label="Over EF-3m (a) / EF-1m (b)"),
    plt.Line2D([], [], ls="", marker="o", color=BIRU, ms=4.5, label="Chronological split"),
    plt.Line2D([], [], ls="", marker="s", color=MERAH, ms=4.5, label="Extreme year withheld")],
    loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
simpan(fig, "gambar_rev4_ekstrem")

# ============================================================ 3. reliabilitas
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, (sas, judul) in zip(axes, SASARAN):
    d0 = pilih(P, sas)
    blok = [("Chronological test years 2020–2025", d0[(d0["skema"] == "S1") & (d0["tahun"] >= 2020)], BIRU),
            ("Extreme years withheld", d0[d0["skema"] == "S2"], MERAH)]
    for lab, dd, c in blok:
        y, p = dd["y"].to_numpy(), dd["xgb"].to_numpy()
        b = np.minimum((p * 10).astype(int), 9)
        ks = [k for k in range(10) if (b == k).sum() >= 20]
        ax.plot([p[b == k].mean() for k in ks], [y[b == k].mean() for k in ks], "o-", color=c, ms=3.5, lw=1.2, label=lab)
    ax.plot([0, 1], [0, 1], color=KELABU_MUDA, lw=0.8, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(judul, loc="left"); ax.set_xlabel("Predicted probability"); ax.grid(True)
axes[0].set_ylabel("Observed frequency"); axes[1].legend(loc="lower right")
simpan(fig, "gambar_rev5_reliabilitas")

# ============================================================ 4. ablasi
LABEL_K = {"api": "Fire history", "musim": "Season", "iklim": "Climate indices (ONI, DMI)", "hujan": "Rainfall",
           "lahan": "Land", "manusia": "Population", "gambut": "Peat", "cuaca": "Fire weather (FWI)"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.5 * len(KELOMPOK) + 0.2), sharey=True)
for ax, (sas, judul) in zip(axes, SASARAN):
    d0 = pilih(A, sas)
    biasa = d0[(d0["skema"] == "S1") & (d0["tahun"] >= 2020)]
    ks = list(KELOMPOK); yk = np.arange(len(ks))
    for off, jenis, c in ((-0.18, "within", BIRU), (0.18, "pooled", MERAH)):
        for i, k in enumerate(ks):
            t = f"tanpa_{k}"
            if jenis == "within":
                v = dalam_tahun(biasa, "lengkap", t); lo, hi = boot_dalam_tahun(biasa, "lengkap", t)
            else:
                v = ap(biasa["y"], biasa["lengkap"]) - ap(biasa["y"], biasa[t]); lo, hi = boot_gabung(biasa, "lengkap", t)
            ax.barh(i + off, v, height=0.34, color=c)
            ax.plot([lo, hi], [i + off] * 2, color=TINTA, lw=0.8)
    ax.axvline(0, color=KELABU, lw=0.8); ax.set_yticks(yk); ax.set_yticklabels([LABEL_K[k] for k in ks])
    ax.invert_yaxis(); ax.set_title(judul, loc="left"); ax.grid(True, axis="x")
    ax.set_xlabel("AUC-PR lost when group removed")
fig.legend(handles=[Patch(color=BIRU, label="Scored within each year"),
                    Patch(color=MERAH, label="Test years pooled")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13))
simpan(fig, "gambar_rev3_ablasi")

# ============================================================ 5. deret nasional
bul = panel.groupby("bulan")["titik_panas"].sum()
x = pd.to_datetime(bul.index)
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.fill_between(x, bul.to_numpy() / 1e3, color=KELABU_MUDA, lw=0)
ax.plot(x, bul.to_numpy() / 1e3, color=KELABU, lw=0.8)
for t in EKSTREM:
    ax.axvspan(pd.Timestamp(f"{t}-01-01"), pd.Timestamp(f"{t}-12-31"), color=MERAH, alpha=0.10, lw=0)
    ax.text(pd.Timestamp(f"{t}-07-01"), bul.max() / 1e3 * 1.03, t, ha="center", va="bottom", fontsize=7.5, color=MERAH)
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31"), ymin=0, ymax=0.04, color=BIRU, lw=0)
ax.text(pd.Timestamp("2022-07-01"), bul.max() / 1e3 * 0.30, "chronological test years 2020–2025",
        ha="center", va="bottom", fontsize=7, color=BIRU)
ax.set_ylabel("Detections per month (thousands)"); ax.grid(True, axis="y")
ax.set_xlim(x.min(), x.max()); ax.set_ylim(top=bul.max() / 1e3 * 1.12)
simpan(fig, "gambar_rev1_deret")
print("selesai.")
