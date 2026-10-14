"""
Percobaan 11: analisis untuk naskah JEPIN versi 4, menjawab telaah panel ketiga
(Telaah/2026-10-14c-JEPIN). Semua pengambilan ulang di sini bergugus TAHUN uji.

  A. Enumerasi eksak 1.716 multiset tahun: selang selisih berpasangan AUC-PR GB - klimatologi;
     rerata pembalikan peringkat ROC-AUC (terhadap sel C) dan AUC-PR (terhadap sel D).
  B. Benih GB: 42 (utama) dan 0-9. Untuk tiap benih, peluang eksak GB di atas klimatologi.
     Juga GB rerata sebelas benih (prediksi dirata-rata) sebagai skor tunggal.
  C. Membuang bulan tanpa kebakaran dengan 92 bulan parah ditahan: tiap ulangan mengambil 7 tahun
     dengan pengembalian, lalu menyimpan seluruh bulan parah dan hanya sebagian bulan tanpa
     kebakaran (100 %, 50 %, 25 %, 8,5 % = setara seimbang). 2.000 ulangan.
  D. Panjang rekaman uji m = 2..28 tahun diambil dari tujuh tahun dengan pengembalian, prevalensi
     asli: juara tetap, GB > klim, peringkat utuh sama, rerata pembalikan. 2.000 ulangan
     (m = 7 juga eksak di bagian A). Gambar Gambar_JEPIN/fig_panjang_rekaman.

Pakai:
    python percobaan11_jepin_v4.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \\
        DL_FIRE_SV-C2_792597/oni.ascii.txt
"""
import os
import sys
import itertools
from math import factorial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])
panel_path, oni_path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(11)
ULANG = 2000

df, ambang = muat(panel_path, oni_path)
TAHUN = list(range(2019, 2026))
ys, raws, ths = [], {}, []
gb_benih = {b: [] for b in range(10)}
for th in TAHUN:
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    for nama, (sl, su) in skor_model(latih, uji).items():
        raws.setdefault(nama, []).append(su)
    Xl, Xu = latih[FITUR].fillna(0.0).to_numpy(), uji[FITUR].fillna(0.0).to_numpy()
    for b in range(10):
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                          random_state=b, verbosity=0)
        m.fit(Xl, latih["y"].to_numpy())
        gb_benih[b].append(m.predict_proba(Xu)[:, 1])
    ys.append(uji["y"].to_numpy()); ths.append(uji["tahun"].to_numpy())
y = np.concatenate(ys); raw = {n: np.concatenate(v) for n, v in raws.items()}; thn = np.concatenate(ths)
gb_benih = {b: np.concatenate(v) for b, v in gb_benih.items()}
NAMA = list(raw)
idx_t = {t: np.where(thn == t)[0] for t in TAHUN}
PAS = list(itertools.combinations(NAMA, 2))


def peringkat(v):
    urut = sorted(NAMA, key=lambda n: -v[n])
    return {n: urut.index(n) + 1 for n in NAMA}


def balik(p, a):
    return sum((p[x] - p[z]) * (a[x] - a[z]) < 0 for x, z in PAS)


acD = peringkat({n: average_precision_score(y, raw[n]) for n in NAMA})
acC = peringkat({n: roc_auc_score(y, raw[n]) for n in NAMA})
MULTI = []
for kombi in itertools.combinations_with_replacement(TAHUN, 7):
    k = [kombi.count(t) for t in TAHUN]
    w = factorial(7) / np.prod([factorial(x) for x in k]) / 7 ** 7
    MULTI.append((w, np.concatenate([np.tile(idx_t[t], c) for t, c in zip(TAHUN, k) if c])))


def wpersentil(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w) / np.sum(w)
    return x[np.searchsorted(c, q)]


# ---------------------------------------------------------------- A
print("=" * 90)
print("A. ENUMERASI EKSAK: selisih berpasangan dan kestabilan per metrik")
print("=" * 90)
dif, wts, bD, bC = [], [], [], []
for w, idx in MULTI:
    vD = {n: average_precision_score(y[idx], raw[n][idx]) for n in NAMA}
    vC = {n: roc_auc_score(y[idx], raw[n][idx]) for n in NAMA}
    dif.append(vD["gradient-boosting"] - vD["klimatologi"]); wts.append(w)
    bD.append(balik(peringkat(vD), acD)); bC.append(balik(peringkat(vC), acC))
dif, wts = np.array(dif), np.array(wts)
d0 = average_precision_score(y, raw["gradient-boosting"]) - average_precision_score(y, raw["klimatologi"])
print(f"  selisih AUC-PR GB - klim: {d0:+.3f}, selang 95% eksak [{wpersentil(dif, wts, 0.025):+.3f}, "
      f"{wpersentil(dif, wts, 0.975):+.3f}], P(selisih > 0) = {100 * np.sum(wts * (dif > 0)):.1f}%")
print(f"  rerata pembalikan AUC-PR vs D {np.sum(wts * np.array(bD)):.2f}; ROC-AUC vs C {np.sum(wts * np.array(bC)):.2f}")

# ---------------------------------------------------------------- B
print("\n" + "=" * 90)
print("B. BENIH GRADIENT BOOSTING (peluang eksak GB > klimatologi)")
print("=" * 90)
klim = raw["klimatologi"]


def p_atas(s):
    return sum(w for w, idx in MULTI if average_precision_score(y[idx], s[idx]) > average_precision_score(y[idx], klim[idx]))


hasil_b = [(42, average_precision_score(y, raw["gradient-boosting"]), p_atas(raw["gradient-boosting"]))]
for b in range(10):
    hasil_b.append((b, average_precision_score(y, gb_benih[b]), p_atas(gb_benih[b])))
for b, a, p in hasil_b:
    print(f"  benih {b:>2}: AUC-PR {a:.3f}  P(GB > klim) {100 * p:.1f}%")
ps = np.array([h[2] for h in hasil_b])
print(f"  rentang P lintas 11 benih: {100 * ps.min():.1f}% - {100 * ps.max():.1f}%, rerata {100 * ps.mean():.1f}%")
rata = np.mean([raw["gradient-boosting"]] + [gb_benih[b] for b in range(10)], axis=0)
print(f"  GB rerata 11 benih: AUC-PR {average_precision_score(y, rata):.3f}, P(GB > klim) {100 * p_atas(rata):.1f}%")

# ---------------------------------------------------------------- C
print("\n" + "=" * 90)
print("C. MEMBUANG BULAN TANPA KEBAKARAN, BULAN PARAH DITAHAN (gugus tahun, 2.000 ulangan)")
print("=" * 90)
FRAKSI = [1.0, 0.5, 0.25, 92 / 1084]
for f in FRAKSI:
    b, juara = [], []
    for _ in range(ULANG):
        idx = np.concatenate([idx_t[t] for t in rng.choice(TAHUN, size=7, replace=True)])
        if y[idx].sum() == 0:
            continue
        p, n = idx[y[idx] == 1], idx[y[idx] == 0]
        n = rng.choice(n, size=max(1, int(round(f * len(n)))), replace=False)
        ii = np.concatenate([p, n])
        r = peringkat({m: average_precision_score(y[ii], raw[m][ii]) for m in NAMA})
        b.append(balik(r, acD)); juara.append(r["gradient-boosting"] == 1)
    b = np.array(b)
    print(f"  bulan tanpa kebakaran disimpan {100 * f:5.1f}%: pembalikan rerata {b.mean():.2f} "
          f"[{np.percentile(b, 2.5):.0f}, {np.percentile(b, 97.5):.0f}]; GB juara {100 * np.mean(juara):.1f}%")

# ---------------------------------------------------------------- D
print("\n" + "=" * 90)
print("D. PANJANG REKAMAN UJI (tahun diambil dengan pengembalian, prevalensi asli, 2.000 ulangan)")
print("=" * 90)
kurva = []
for m in (2, 3, 4, 5, 7, 10, 14, 21, 28):
    b, juara, sama, npos = [], [], [], []
    for _ in range(ULANG):
        idx = np.concatenate([idx_t[t] for t in rng.choice(TAHUN, size=m, replace=True)])
        if y[idx].sum() == 0:
            continue
        r = peringkat({n: average_precision_score(y[idx], raw[n][idx]) for n in NAMA})
        bb = balik(r, acD)
        b.append(bb); juara.append(r["gradient-boosting"] == 1); sama.append(bb == 0); npos.append(y[idx].sum())
    kurva.append((m, np.mean(npos), 100 * np.mean(juara), 100 * np.mean(sama), np.mean(b)))
    print(f"  {m:>2} tahun (rerata {np.mean(npos):5.1f} bulan parah): GB juara {100 * np.mean(juara):5.1f}%; "
          f"peringkat utuh sama {100 * np.mean(sama):5.1f}%; rerata pembalikan {np.mean(b):.2f}")

# ---------------------------------------------------------------- gambar
BIRU, TINTA, KELABU = "#1F5FA8", "#1F2328", "#6B7280"
plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.edgecolor": "#B8BCC4", "axes.labelcolor": TINTA,
                     "xtick.color": "#5A6169", "ytick.color": "#5A6169", "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "grid.color": "#E7E9EC"})
k = np.array(kurva)
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(k[:, 0], k[:, 2], "-o", color=BIRU, lw=2, ms=6)
ax.plot(k[:, 0], k[:, 3], "--o", color=KELABU, lw=2, ms=6)
ax.annotate("gradient boosting ranked first", (k[-1, 0], k[-1, 2]), xytext=(-4, 8),
            textcoords="offset points", ha="right", color=BIRU, fontsize=8)
ax.annotate("full seven-model ranking reproduced", (k[-1, 0], k[-1, 3]), xytext=(-4, -14),
            textcoords="offset points", ha="right", color=KELABU, fontsize=8)
ax.axvline(7, color="#B8BCC4", lw=1, zorder=0)
ax.text(7.3, 3, "observed\nrecord", fontsize=7, color=KELABU, va="bottom")
ax.set_xticks([2, 4, 7, 10, 14, 21, 28])
ax.set_xlabel("Test record length (years drawn from the seven observed)")
ax.set_ylabel("Share of resamples (%)")
ax.set_ylim(0, 100)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y")
ax.set_axisbelow(True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(BASE, "Gambar_JEPIN", f"fig_panjang_rekaman.{ext}"))
print("\n  tersimpan: Gambar_JEPIN/fig_panjang_rekaman.png dan .pdf")
print("\nselesai.")
