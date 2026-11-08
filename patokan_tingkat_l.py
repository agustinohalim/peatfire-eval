"""
Revisi fase L (Hasil_Patokan_17): agregasi tahun untuk hasil peluang VIIRS (Tabel 4 naskah dan bagian tingkat),
sesudah panel 13. Data, prediksi tersimpan (fase E), kolom dan pemecahan geser sama dengan patokan_tingkat_i.py;
semua kabupaten, enam tahun biasa 2020-2025 (skema S1).
  L1  sel Tabel 4: BSS (a) rata-rata rasio tahunan (= fase I) dua tahap + t95; (b) satu rasio atas semua baris uji
      (1 - jumlah galat kuadrat model / jumlah rujukan), dua tahap; (c) median rasio tahunan dan banyak tahun > 0
  L2  bagian tingkat GB/RF atas kalender dan EF-P: total (sesudah geser per tahun) dan offset periode, dihitung
      sebagai satu rasio (bagian = 1 - BSS sesudah geser / BSS mentah, keduanya satu rasio), dua tahap

Pakai (satu sasaran per jalan):
    python -u patokan_tingkat_l.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_l_kab.log
    python -u patokan_tingkat_l.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_l_nas.log
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
REF = B.PASANGAN[SAS]
N = 1000
KUNCI = ["sasaran", "skema", "tag", "gid", "bulan"]

P = B.muat_P(fwi=False)
P = P[(P.sasaran == SAS) & (P.skema == "S1") & P.tag.isin(B.BIASA)]
for f, kol in ((f"patokan_revisi_peluang_{SAS}.csv", ["konstan", "rujukan_peluang", "ef1_tingkat"]),
               ("patokan_revisi_sederhana.csv", ["log_ef1_jeda"]), ("patokan_revisi_setel.csv", ["xgb_setel"]),
               (f"patokan_revisi_tingkat_gb_{SAS}.csv", ["xgb_nas"])):
    X = pd.read_csv(os.path.join(B.D, f), dtype={"tag": str})
    X = X[(X.sasaran == SAS) & (X.skema == "S1") & X.tag.isin(B.BIASA)][KUNCI + kol]
    P = P.merge(X, on=KUNCI, how="left", validate="one_to_one")
g = B.ambil_sasaran(P, SAS).copy().reset_index(drop=True)
del P
MODEL = ["log_ef1_jeda", "rf", "xgb", "xgb_setel", "mlp"]
KOL = [REF, "konstan", "rujukan_peluang", "ef1_tingkat"] + MODEL + ["xgb_nas"]
assert not g[KOL].isna().any().any(), g[KOL].isna().sum()
print(f"{SAS}: rujukan pasangan {B.LABEL[REF]}; kabupaten {g.gid.nunique()}; baris {len(g):,}; tahun {', '.join(B.BIASA)}; "
      f"prevalensi per tahun " + ", ".join(f"{t} {x.y.mean() * 100:.2f}%" for t, x in g.groupby("tag")), flush=True)


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def geser(f, ybar):
    lf = logit(f)
    h = lambda dl: np.mean(1 / (1 + np.exp(-(lf + dl)))) - ybar
    return 1 / (1 + np.exp(-(lf + brentq(h, -30, 30))))


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


PEC = ["xgb", "rf", REF, "rujukan_peluang"]
for c in PEC:
    g[c + "_pd"] = geser(g[c].to_numpy(), g.y.mean())
    g[c + "_lv"] = np.nan
    for _, x in g.groupby("tag"):
        g.loc[x.index, c + "_lv"] = geser(x[c].to_numpy(), x.y.mean())
SEMUA = KOL + [c + w for c in PEC for w in ("_pd", "_lv")]
ix = {c: i for i, c in enumerate(SEMUA)}
kab = g.gid.unique()
blok = {t: B.Blok(x, SEMUA, kab) for t, x in g.groupby("tag")}


def fn(b, j):
    y = b.y[j]
    return np.array([np.sum((b.s[c][j] - y) ** 2) for c in SEMUA])


tags = list(B.BIASA)
th = np.array([fn(blok[t], slice(None)) for t in tags])
R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)


def rata(a, r):
    return 1 - th[:, ix[a]] / th[:, ix[r]], (1 - R2[:, :, ix[a]] / R2[:, :, ix[r]]).mean(1)


def satu(a, r):
    return 1 - th[:, ix[a]].sum() / th[:, ix[r]].sum(), 1 - R2[:, :, ix[a]].sum(1) / R2[:, :, ix[r]].sum(1)


print("\nL1 Tabel 4: (a) rata-rata rasio [2 tahap] t95; (b) satu rasio [2 tahap]; (c) median, tahun > 0 (n = 6)")
for acu in (REF, "rujukan_peluang", "ef1_tingkat"):
    print(f"  -- atas {acu}")
    for c in (["konstan", "rujukan_peluang", "ef1_tingkat"] if acu == REF else []) + MODEL + (["xgb_nas"] if acu != "rujukan_peluang" else []):
        pt, ds = rata(c, acu)
        lo, hi = np.percentile(ds, [2.5, 97.5]); tl, tu = t95(pt)
        sr, dsr = satu(c, acu)
        slo, shi = np.percentile(dsr, [2.5, 97.5])
        a2 = lo > 0 or hi < 0; s2 = slo > 0 or shi < 0
        print(f"    {c:<16}(a) {pt.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]  "
              f"(b) {sr:+.4f} [{slo:+.4f}, {shi:+.4f}]  (c) {np.median(pt):+.4f} {int((pt > 0).sum())}/6"
              f"{'  TANDA BERBALIK' if np.sign(sr) != np.sign(pt.mean()) else ''}"
              f"{'  VONIS 2TAHAP BEDA' if a2 != s2 else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)

print("\nL2 Bagian tingkat sebagai satu rasio: mentah / sesudah offset periode / sesudah geser per tahun; bagian [2 tahap]")
for acu in (REF, "rujukan_peluang"):
    for c in ("xgb", "rf"):
        raw, rd = satu(c, acu)
        p_, pdd = satu(c + "_pd", acu + "_pd")
        lv, lvd = satu(c + "_lv", acu + "_lv")
        bp, bt, bto = 1 - pdd / rd, (pdd - lvd) / rd, 1 - lvd / rd
        q = lambda d: f"[{np.percentile(d, 2.5):.2f}, {np.percentile(d, 97.5):.2f}]"
        print(f"    {c:<4} atas {acu:<16} mentah {raw:+.4f}  periode {p_:+.4f}  tahun {lv:+.4f}  "
              f"bagian periode {1 - p_ / raw:.2f} {q(bp)}  bagian tahun {(p_ - lv) / raw:.2f} {q(bt)}  "
              f"total {1 - lv / raw:.2f} {q(bto)}", flush=True)
print("selesai.")
