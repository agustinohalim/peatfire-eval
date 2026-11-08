"""
Revisi fase J (Hasil_Patokan_15), J3: seperti patokan_tingkat_i.py, tetapi hanya kabupaten bergambut
(gambut_fraksi > 0 dari gambut_kabupaten.csv, strata yang sama dengan Suplemen S7); geser tingkat dihitung
di dalam strata (laju teramati kabupaten bergambut). Panel VIIRS, tahun biasa 2020-2025 (skema S1):
  I4  pemecahan geser-tingkat: BSS mentah / sesudah offset periode (satu geser logit untuk semua baris uji
      semua tahun) / sesudah geser per tahun (fase E); bagian periode dan bagian tahun, dua tahap.
  I5  sel Tabel 4 naskah: BSS, dua tahap, t95 (df 5), MDD80, dan tanda bila vonis dua selang berbeda.

Pakai (satu sasaran per jalan):
    python -u patokan_tingkat_j.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_j_kab.log
    python -u patokan_tingkat_j.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_j_nas.log
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
g = B.ambil_sasaran(P, SAS).copy()
del P
GB_ = pd.read_csv(os.path.join(B.D, "gambut_kabupaten.csv"))
g["gambut"] = g.gid.map(GB_.set_index("gid").gambut_fraksi).fillna(0.0)
print(f"semua kabupaten {g.gid.nunique()}, baris {len(g):,}", flush=True)
g = g[g.gambut > 0].reset_index(drop=True)
print(f"STRATA bergambut (> 0): kabupaten {g.gid.nunique()}, baris {len(g):,}, prevalensi {g.y.mean() * 100:.2f}%", flush=True)
MODEL = ["log_ef1_jeda", "rf", "xgb", "xgb_setel", "mlp"]
KOL = [REF, "konstan", "rujukan_peluang", "ef1_tingkat"] + MODEL + ["xgb_nas"]
assert not g[KOL].isna().any().any(), g[KOL].isna().sum()
print(f"{SAS}: rujukan pasangan {B.LABEL[REF]}; baris {len(g):,}; tahun {', '.join(B.BIASA)}", flush=True)


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


def mdd80(pt):
    n = len(pt)
    return (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * pt.std(ddof=1) / np.sqrt(n)


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
    return np.array([np.mean((b.s[c][j] - y) ** 2) for c in SEMUA])


tags = list(B.BIASA)
th = np.array([fn(blok[t], slice(None)) for t in tags])
R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)


def bss(a, r):
    return 1 - th[:, ix[a]] / th[:, ix[r]], (1 - R2[:, :, ix[a]] / R2[:, :, ix[r]]).mean(1)


print("\nI5 Tabel 4: BSS 2tahap t95 MDD80 (n = 6, df 5); BEDA = satu selang memuat nol, yang lain tidak")
for acu in (REF, "rujukan_peluang", "ef1_tingkat"):
    print(f"  -- atas {acu}")
    for c in (["konstan", "rujukan_peluang", "ef1_tingkat"] if acu == REF else []) + MODEL + (["xgb_nas"] if acu != "rujukan_peluang" else []):
        pt, ds = bss(c, acu)
        lo, hi = np.percentile(ds, [2.5, 97.5]); tl, tu = t95(pt)
        a, b = lo > 0 or hi < 0, tl > 0 or tu < 0
        print(f"    {c:<16}{pt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}] MDD80 {mdd80(pt):.4f}"
              f"{'  KEDUANYA' if a and b else ('  BEDA' if a != b else '')}", flush=True)

print("\nI4 Pemecahan geser-tingkat: mentah / offset periode / per tahun; bagian periode, bagian tahun [2 tahap]")
for acu in (REF, "rujukan_peluang"):
    for c in ("xgb", "rf"):
        raw, rd = bss(c, acu)
        p_, pdd = bss(c + "_pd", acu + "_pd")
        lv, lvd = bss(c + "_lv", acu + "_lv")
        bp, bt = 1 - pdd / rd, (pdd - lvd) / rd
        tl, tu = t95(p_)
        print(f"    {c:<4} atas {acu:<16} mentah {raw.mean():+.4f}  periode {p_.mean():+.4f} "
              f"[{np.percentile(pdd, 2.5):+.4f}, {np.percentile(pdd, 97.5):+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]  "
              f"tahun {lv.mean():+.4f}  bagian periode {1 - p_.mean() / raw.mean():.2f} "
              f"[{np.percentile(bp, 2.5):.2f}, {np.percentile(bp, 97.5):.2f}]  bagian tahun "
              f"{(p_.mean() - lv.mean()) / raw.mean():.2f} [{np.percentile(bt, 2.5):.2f}, {np.percentile(bt, 97.5):.2f}]  "
              f"total {1 - lv.mean() / raw.mean():.2f}", flush=True)
print("  geser periode (logit) " + "  ".join(
    f"{c} {brentq(lambda dl: np.mean(1 / (1 + np.exp(-(logit(g[c].to_numpy()) + dl)))) - g.y.mean(), -30, 30):+.3f}" for c in PEC))
print("selesai.")
