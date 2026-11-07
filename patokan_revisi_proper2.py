"""
PeatFireBench revisi fase D3d (Hasil_Patokan_9) — TAMBAHAN PASCA-HOC, ditulis sesudah melihat D3a:
EF-1m mentah adalah peluang kasar (frekuensi dari <= 12 tahun latih), sehingga BSS atas EF-1m bisa
mencerminkan kekasaran rujukan, bukan keunggulan model. Di sini BSS dan selisih log score rf, xgb, mlp
atas rujukan peluang yang lebih halus: logistik EF-1m + jeda (log_ef1_jeda) dan EF-3m, kedua sasaran,
tahun biasa (dua tahap + t) dan ekstrem (per tahun).

Pakai:    python -u patokan_revisi_proper2.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_proper2.log
"""
import numpy as np
import pandas as pd
from scipy import stats

import patokan_revisi_bantu as B

EPS = 1e-4
P = B.muat_P(fwi=False)
S = pd.read_csv(f"{B.D}/patokan_revisi_sederhana.csv", dtype={"tag": str})
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = P.merge(S[kunci + ["log_ef1_jeda"]], on=kunci, how="left", validate="one_to_one")
MODEL = ["rf", "xgb", "mlp", "log_ef1_jeda"]


def bs(y, p):
    return np.mean((p - y) ** 2)


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


print("=" * 100 + "\nD3d (pasca-hoc). BSS dan selisih log score atas rujukan peluang yang lebih halus\n" + "=" * 100)
for sas in B.PASANGAN:
    d = B.ambil_sasaran(P, sas)
    kol = MODEL + ["klim_frek", "klim_frek1"]
    for kel, sk, tags in (("biasa", "S1", B.BIASA), ("ekstrem", "S2", B.EKSTREM)):
        g = d[(d.skema == sk) & d.tag.isin(tags)]
        kab = g.gid.unique()
        blok = {t: B.Blok(x, kol, kab) for t, x in g.groupby("tag")}

        def fn(b, j):
            y = b.y[j]
            return np.r_[[bs(y, b.s[m][j]) for m in kol], [ll(y, b.s[m][j]) for m in kol]]

        th = np.array([fn(blok[t], slice(None)) for t in tags]); k = len(kol); ix = {m: i for i, m in enumerate(kol)}
        print(f"\n--- {sas}, {kel}: Brier rata-rata " + "  ".join(f"{m} {th[:, i].mean():.5f}" for m, i in ix.items()))
        if kel == "biasa":
            R = B.dua_tahap(blok, fn, 1000, np.random.default_rng(B.SEED), kab)
        for ref in ("log_ef1_jeda", "klim_frek"):
            for m in ("rf", "xgb", "mlp") + (("log_ef1_jeda",) if ref == "klim_frek" else ()):
                i, r = ix[m], ix[ref]
                bt = 1 - th[:, i] / th[:, r]; dl = th[:, k + r] - th[:, k + i]
                nama = f"{m} atas {B.LABEL.get(ref, ref)}"
                if kel == "biasa":
                    lo, hi = np.percentile((1 - R[:, :, i] / R[:, :, r]).mean(1), [2.5, 97.5]); tl, tu = t95(bt)
                    lo2, hi2 = np.percentile((R[:, :, k + r] - R[:, :, k + i]).mean(1), [2.5, 97.5]); tl2, tu2 = t95(dl)
                    print(f"    BSS {nama:<28}{bt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
                          f"{'  LOLOS R16' if lo > 0 and tl > 0 else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in bt))
                    print(f"    dlog {nama:<27}{dl.mean():+.4f} 2tahap [{lo2:+.4f}, {hi2:+.4f}] t95 [{tl2:+.4f}, {tu2:+.4f}]"
                          f"{'  LOLOS R16' if lo2 > 0 and tl2 > 0 else ''}", flush=True)
                else:
                    print(f"    BSS {nama:<28}" + "  ".join(f"{t} {x:+.3f}" for t, x in zip(tags, bt))
                          + "   dlog " + "  ".join(f"{t} {x:+.4f}" for t, x in zip(tags, dl)))
print("\nselesai.")
