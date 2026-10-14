"""
Percobaan 8: ketidakpastian peringkat model di panel Kalimantan Barat — untuk naskah JEPIN.

Sesudah percobaan 7 menunjukkan bahwa goyahnya peringkat berasal dari sedikitnya positif dan
ukuran data uji, bukan dari penyeimbangan, naskah JEPIN dibingkai ulang (14 Oktober 2026):
seberapa pasti peringkat tujuh model pada data satu provinsi, dan apa yang perlu dilaporkan.

  A. AUC-PR dan ROC-AUC tiap model dengan selang bootstrap gugus tahun (7 tahun uji)
  B. Peluang tiap model menduduki tiap peringkat AUC-PR (bootstrap gugus tahun dan iid)
  C. Peluang model i di atas model j untuk pasangan bertetangga dalam peringkat sel D
  D. Pembalikan lawan jumlah positif: bootstrap berstrata prevalensi asli dan seimbang
     pada 23, 46, 69, 92 positif
  E. Benih gradient boosting 0-9: AUC-PR dan peringkatnya
  F. Skill ternormalisasi per tahun uji 2019-2025, semua model (untuk gambar)
  G. Peringkat bila ambang 90 persen dihitung dari tahun latih saja (per lipatan)

Gambar ke Gambar_JEPIN/: peluang peringkat, kurva jumlah positif, skill per tahun.

Pakai:
    python percobaan8_jepin.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \\
        DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
ULANG = 1000
rng = np.random.default_rng(8)
OUT = os.path.join(BASE, "Gambar_JEPIN")
os.makedirs(OUT, exist_ok=True)

LABEL = {"klimatologi": "Climatology", "gradient-boosting": "Gradient boosting",
         "persistence": "Persistence", "seasonal-naive": "Seasonal naive",
         "rasio-analog": "Ratio scaling", "regresi-ONI": "Logistic, ONI",
         "regresi-penuh": "Logistic, full"}
BIRU, MERAH = "#1F5FA8", "#C1442A"
KELABU, KELABU_MUDA = "#6B7280", "#B8BCC4"
TINTA, TINTA_2 = "#1F2328", "#5A6169"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": KELABU_MUDA, "axes.linewidth": 0.8, "axes.labelcolor": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": "#E7E9EC", "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8,
})


def bersih(ax, kisi="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=kisi, zorder=0)
    ax.set_axisbelow(True)


def simpan(fig, nama):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nama}.{ext}"))
    plt.close(fig)
    print(f"  tersimpan: Gambar_JEPIN/{nama}.png dan .pdf")


# ---------------------------------------------------------------- skor

df, ambang = muat(panel_path, oni_path)
kum, thn, kab = {}, [], []
for th in range(2019, 2026):
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    thn.append(uji["tahun"].to_numpy()); kab.append(uji["kabupaten"].to_numpy())
    for nama, (sl, su) in skor_model(latih, uji).items():
        kum.setdefault(nama, {"y": [], "raw": []})
        kum[nama]["y"].append(uji["y"].to_numpy())
        kum[nama]["raw"].append(su)
NAMA = list(kum)
y = np.concatenate(kum[NAMA[0]]["y"])
raw = {n: np.concatenate(kum[n]["raw"]) for n in NAMA}
thn, kab = np.concatenate(thn), np.concatenate(kab)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
PASANGAN = list(itertools.combinations(NAMA, 2))
thn_u = np.unique(thn)
idx_thn = {t: np.where(thn == t)[0] for t in thn_u}


def peringkat(nilai):
    urut = sorted(NAMA, key=lambda n: -nilai[n])
    return {n: urut.index(n) + 1 for n in NAMA}


def balik(p, acuan):
    return sum((p[a] - p[b]) * (acuan[a] - acuan[b]) < 0 for a, b in PASANGAN)


nilaiD = {n: average_precision_score(y, raw[n]) for n in NAMA}
nilaiC = {n: roc_auc_score(y, raw[n]) for n in NAMA}
acuanD, acuanC = peringkat(nilaiD), peringkat(nilaiC)
URUT = sorted(NAMA, key=lambda n: acuanD[n])

print(f"Panel: {len(df)} baris, ambang > {ambang:.0f}, uji {len(y)}, positif {len(pos)} "
      f"({y.mean() * 100:.1f}%), {ULANG} ulangan bootstrap")


def ambil_tahun():
    return np.concatenate([idx_thn[t] for t in rng.choice(thn_u, size=len(thn_u))])


def ambil_iid():
    return rng.choice(len(y), size=len(y), replace=True)


# ---------------------------------------------------------------- A, B, C

hasil = {}
for skema, fungsi in [("tahun", ambil_tahun), ("iid", ambil_iid)]:
    pr, roc, rank = {n: [] for n in NAMA}, {n: [] for n in NAMA}, []
    for _ in range(ULANG):
        while True:
            idx = fungsi()
            if 0 < y[idx].sum() < len(idx):
                break
        v = {n: average_precision_score(y[idx], raw[n][idx]) for n in NAMA}
        for n in NAMA:
            pr[n].append(v[n]); roc[n].append(roc_auc_score(y[idx], raw[n][idx]))
        rank.append(peringkat(v))
    hasil[skema] = (pr, roc, rank)

print("\n" + "=" * 92)
print("A. METRIK DENGAN SELANG 95% (bootstrap gugus tahun, 7 tahun uji; iid dalam kurung)")
print("=" * 92)
pr_t, roc_t, rank_t = hasil["tahun"]
pr_i, roc_i, rank_i = hasil["iid"]
for n in URUT:
    q = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
    print(f"  {LABEL[n]:<18} AUC-PR {nilaiD[n]:.3f} {q(pr_t[n])} ({q(pr_i[n])})   "
          f"ROC-AUC {nilaiC[n]:.3f} {q(roc_t[n])}   peringkat D {acuanD[n]}, C {acuanC[n]}")

print("\n" + "=" * 92)
print("B. PELUANG PERINGKAT AUC-PR (%), bootstrap gugus tahun | iid")
print("=" * 92)
P = {}
for skema in ("tahun", "iid"):
    rk = hasil[skema][2]
    P[skema] = np.array([[np.mean([r[n] == k for r in rk]) for k in range(1, 8)] for n in URUT])
print(f"  {'':<18}" + "".join(f"{k:>7}" for k in range(1, 8)))
for i, n in enumerate(URUT):
    print(f"  {LABEL[n]:<18}" + "".join(f"{100 * P['tahun'][i, k]:>7.0f}" for k in range(7))
          + "   |" + "".join(f"{100 * P['iid'][i, k]:>5.0f}" for k in range(7)))
for skema in ("tahun", "iid"):
    rk = hasil[skema][2]
    b = np.array([balik(r, acuanD) for r in rk])
    sama = np.mean([all(r[n] == acuanD[n] for n in NAMA) for r in rk])
    print(f"  {skema}: pembalikan lawan D median {np.median(b):.0f} [{np.percentile(b, 2.5):.0f}, "
          f"{np.percentile(b, 97.5):.0f}], rerata {b.mean():.2f}; peringkat utuh sama {100 * sama:.1f}%")

print("\n" + "=" * 92)
print("C. PELUANG MODEL DI ATAS TETANGGANYA (AUC-PR, gugus tahun | iid)")
print("=" * 92)
for a, b in zip(URUT[:-1], URUT[1:]):
    pt = np.mean(np.array(pr_t[a]) > np.array(pr_t[b]))
    pi = np.mean(np.array(pr_i[a]) > np.array(pr_i[b]))
    d = nilaiD[a] - nilaiD[b]
    print(f"  {LABEL[a]:<18} > {LABEL[b]:<18} selisih {d:+.3f}   P {100 * pt:5.1f}% | {100 * pi:5.1f}%")

# ---------------------------------------------------------------- D

print("\n" + "=" * 92)
print("D. PEMBALIKAN LAWAN JUMLAH POSITIF (300 ulangan; acuan = data uji utuh)")
print("=" * 92)
kurva = {"asli": [], "seimbang": []}
TINGKAT = [23, 46, 69, 92]
for p in TINGKAT:
    nn = int(round(p * len(neg) / len(pos)))
    for skema in ("asli", "seimbang"):
        bD, bC, juara = [], [], []
        for _ in range(300):
            ip = rng.choice(pos, size=p, replace=True)
            ineg = rng.choice(neg, size=nn if skema == "asli" else p, replace=True)
            idx = np.concatenate([ip, ineg])
            rD = peringkat({n: average_precision_score(y[idx], raw[n][idx]) for n in NAMA})
            rC = peringkat({n: roc_auc_score(y[idx], raw[n][idx]) for n in NAMA})
            bD.append(balik(rD, acuanD)); bC.append(balik(rC, acuanC))
            juara.append(rD[URUT[0]] == 1)
        bD, bC = np.array(bD), np.array(bC)
        kurva[skema].append((p, bD.mean(), np.percentile(bD, 2.5), np.percentile(bD, 97.5), bC.mean()))
        print(f"  {p:>3} positif, {skema:<9} n={p + (nn if skema == 'asli' else p):>5}: "
              f"AUC-PR rerata {bD.mean():.2f} [{np.percentile(bD, 2.5):.0f}, {np.percentile(bD, 97.5):.0f}]"
              f"  ROC-AUC rerata {bC.mean():.2f}  juara D tetap {100 * np.mean(juara):.0f}%")

# ---------------------------------------------------------------- E

print("\n" + "=" * 92)
print("E. BENIH GRADIENT BOOSTING (random_state 0-9)")
print("=" * 92)
Xcols = FITUR
gb_pr = []
for benih in range(10):
    su_all = []
    for th in range(2019, 2026):
        latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                          random_state=benih, verbosity=0)
        m.fit(latih[Xcols].fillna(0.0).to_numpy(), latih["y"].to_numpy())
        su_all.append(m.predict_proba(uji[Xcols].fillna(0.0).to_numpy())[:, 1])
    s = np.concatenate(su_all)
    v = dict(nilaiD); v["gradient-boosting"] = average_precision_score(y, s)
    gb_pr.append((v["gradient-boosting"], peringkat(v)["gradient-boosting"]))
a = np.array([g[0] for g in gb_pr])
print(f"  AUC-PR GB lintas 10 benih: min {a.min():.3f}, maks {a.max():.3f}, rerata {a.mean():.3f}; "
      f"peringkat: {sorted(set(g[1] for g in gb_pr))}")

# ---------------------------------------------------------------- F

print("\n" + "=" * 92)
print("F. SKILL TERNORMALISASI PER TAHUN UJI")
print("=" * 92)
skill = {n: [] for n in NAMA}
for t in thn_u:
    i = idx_thn[t]; prev = y[i].mean()
    for n in NAMA:
        skill[n].append((average_precision_score(y[i], raw[n][i]) - prev) / (1 - prev))
print(f"  {'':<18}" + "".join(f"{t:>8}" for t in thn_u))
for n in URUT:
    print(f"  {LABEL[n]:<18}" + "".join(f"{s:>8.3f}" for s in skill[n]))

# ---------------------------------------------------------------- G

print("\n" + "=" * 92)
print("G. AMBANG DARI TAHUN LATIH SAJA (persentil 90 data latih tiap lipatan)")
print("=" * 92)
yG, rawG = [], {n: [] for n in NAMA}
for th in range(2019, 2026):
    d2 = df.copy()
    amb = d2.loc[d2["tahun"] < th, "titik_panas"].quantile(0.90)
    d2["y"] = (d2["titik_panas"] > amb).astype(int)
    latih, uji = d2[d2["tahun"] < th], d2[d2["tahun"] == th]
    for nama, (sl, su) in skor_model(latih, uji).items():
        rawG[nama].append(su)
    yG.append(uji["y"].to_numpy())
yG = np.concatenate(yG); rawG = {n: np.concatenate(rawG[n]) for n in NAMA}
vG = {n: average_precision_score(yG, rawG[n]) for n in NAMA}
rG = peringkat(vG)
print(f"  positif {yG.sum()} ({100 * yG.mean():.1f}%)")
for n in sorted(NAMA, key=lambda n: rG[n]):
    print(f"  {LABEL[n]:<18} AUC-PR {vG[n]:.3f}  peringkat {rG[n]} (ambang penuh: {acuanD[n]})")
print(f"  pembalikan lawan peringkat ambang penuh: {balik(rG, acuanD)} dari 21")

# ---------------------------------------------------------------- gambar

print("\nGambar")
# 1. peluang peringkat, gugus tahun
cmap = LinearSegmentedColormap.from_list("biru", ["#FFFFFF", "#DCE7F5", "#8FB0DB", BIRU, "#0F3666"])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
M = 100 * P["tahun"]
ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for k in range(M.shape[1]):
        if M[i, k] >= 0.5:
            ax.text(k, i, f"{M[i, k]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, k] > 55 else TINTA)
ax.set_xticks(range(7), [str(k) for k in range(1, 8)])
ax.set_yticks(range(7), [LABEL[n] for n in URUT])
ax.set_xlabel("Rank by AUC-PR (1 = best)")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
simpan(fig, "fig_peluang_peringkat")

# 2. pembalikan lawan jumlah positif
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for skema, warna, gaya, label in [("asli", TINTA, "-", "natural prevalence"),
                                  ("seimbang", KELABU, "--", "balanced")]:
    k = np.array(kurva[skema])
    ax.fill_between(k[:, 0], k[:, 2], k[:, 3], color=warna, alpha=0.10, lw=0)
    ax.plot(k[:, 0], k[:, 1], gaya, color=warna, lw=2, marker="o", ms=6)
    ax.annotate(label, (k[0, 0], k[0, 1]), xytext=(6, 6), textcoords="offset points",
                color=warna, fontsize=8)
ax.set_xticks(TINGKAT)
ax.set_xlabel("Severe district-months in the test set")
ax.set_ylabel("Reversed model pairs (of 21), AUC-PR")
ax.set_ylim(bottom=0)
bersih(ax)
simpan(fig, "fig_jumlah_positif")

# 3. skill per tahun
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for n in NAMA:
    if n in ("klimatologi", "gradient-boosting"):
        continue
    ax.plot(thn_u, skill[n], color=KELABU_MUDA, lw=1, zorder=2)
for n, w in [("klimatologi", MERAH), ("gradient-boosting", BIRU)]:
    ax.plot(thn_u, skill[n], color=w, lw=2, marker="o", ms=6, zorder=3, label=LABEL[n])
ax.plot([], [], color=KELABU_MUDA, lw=1, label="five other models")
ax.set_xticks(thn_u)
ax.set_xlabel("Test year")
ax.set_ylabel("Normalised AUC-PR skill")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
bersih(ax)
simpan(fig, "fig_skill_per_tahun")
print("\nselesai.")
