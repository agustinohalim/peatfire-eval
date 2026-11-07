"""
PeatFireBench revisi fase D3 (Hasil_Patokan_9): skor proper. Brier skill score dan selisih log score atas
rujukan pasangan (mentah), BSS dengan model dan rujukan dikalibrasi ulang (Platt dari luar-lipatan, per
kabupaten), dekomposisi Murphy (10 bin berjumlah sama), dan kalibrasi dengan selang (calibration-in-the-
large, slope, Brier) untuk xgb mentah / Platt semua tahun / Platt pasca-2015.

Pakai:    python -u patokan_revisi_proper.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_proper.log
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
MODEL = ["rf", "xgb", "mlp", "log_ef1_jeda"]
EPS = 1e-4
N_ULANG = 1000
P = B.muat_P(fwi=False)
S = pd.read_csv(f"{B.D}/patokan_revisi_sederhana.csv", dtype={"tag": str})
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = P.merge(S[kunci + ["log_ef1_jeda"]], on=kunci, how="left", validate="one_to_one")
O = pd.read_csv(f"{B.D}/patokan_revisi_oof.csv", dtype={"tag": str})
O = O[O.ada_positif == 1]


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def platt(p_latih, y_latih):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_latih)[:, None], y_latih)
    return lambda p: m.predict_proba(logit(p)[:, None])[:, 1]


def bs(y, p):
    return np.mean((p - y) ** 2)


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def murphy(y, p, nb=10):
    o = np.argsort(p, kind="stable"); ys, ps = y[o], p[o]
    rel = res = 0.0; ob = y.mean()
    for k in np.array_split(np.arange(len(y)), nb):
        if len(k) == 0:
            continue
        rel += len(k) * (ps[k].mean() - ys[k].mean()) ** 2; res += len(k) * (ys[k].mean() - ob) ** 2
    return rel / len(y), res / len(y), ob * (1 - ob)


def slope(y, p):
    if y.sum() == 0:
        return np.nan
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y).coef_[0, 0]


def cetak(nama, titik, dist, extra=""):
    lo, hi = np.nanpercentile(dist, [2.5, 97.5])
    print(f"    {nama:<40}{titik:+.4f} [{lo:+.4f}, {hi:+.4f}]{extra}", flush=True)


def t_tahun(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return f"  t95 [{pt.mean() - q * se:+.4f}, {pt.mean() + q * se:+.4f}]"


print("=" * 100 + "\nD3a. Brier skill score dan selisih log score atas rujukan pasangan (mentah)\n" + "=" * 100)
for sas, ref in B.PASANGAN.items():
    d = B.ambil_sasaran(P, sas)
    kol = MODEL + [ref]
    for kel, sk, tags in (("biasa", "S1", B.BIASA), ("ekstrem", "S2", B.EKSTREM)):
        g = d[(d.skema == sk) & d.tag.isin(tags)]
        assert g[kol].notna().all().all()
        kab = g.gid.unique()
        blok = {t: B.Blok(x, kol, kab) for t, x in g.groupby("tag")}

        def fn(b, j):
            y = b.y[j]
            return np.r_[[bs(y, b.s[m][j]) for m in kol], [ll(y, b.s[m][j]) for m in kol]]

        th = np.array([fn(blok[t], slice(None)) for t in tags])
        k = len(kol); ir = k - 1
        print(f"\n--- {sas}, {kel}, rujukan {B.LABEL[ref]}: Brier rujukan per tahun "
              + " ".join(f"{t} {x:.4f}" for t, x in zip(tags, th[:, ir])))
        if kel == "biasa":
            R = B.dua_tahap(blok, fn, N_ULANG, np.random.default_rng(B.SEED), kab)
        for i, m in enumerate(MODEL):
            bss_th = 1 - th[:, i] / th[:, ir]; dll_th = th[:, k + ir] - th[:, k + i]
            w = np.array([blok[t].y.sum() for t in tags], float)
            if kel == "biasa":
                bss_R = (1 - R[:, :, i] / R[:, :, ir]).mean(1); dll_R = (R[:, :, k + ir] - R[:, :, k + i]).mean(1)
                cetak(f"BSS {m}", bss_th.mean(), bss_R, t_tahun(bss_th) + "  per tahun " + " ".join(f"{x:+.3f}" for x in bss_th)
                      + f"  bobot kejadian {(w * bss_th).sum() / w.sum():+.4f}")
                cetak(f"dlogscore {m}", dll_th.mean(), dll_R, t_tahun(dll_th) + "  per tahun " + " ".join(f"{x:+.4f}" for x in dll_th))
            else:
                print(f"    BSS {m:<36}" + "  ".join(f"{t} {x:+.3f}" for t, x in zip(tags, bss_th))
                      + "   dlogscore " + "  ".join(f"{t} {x:+.4f}" for t, x in zip(tags, dll_th)))
        # dekomposisi Murphy
        for m in ("xgb", "rf", ref):
            rr = np.array([murphy(blok[t].y, blok[t].s[m]) for t in tags])
            print(f"    Murphy {m:<12} REL {rr[:, 0].mean():.5f}  RES {rr[:, 1].mean():.5f}  UNC {rr[:, 2].mean():.5f}")

print("\n" + "=" * 100 + "\nD3b. BSS dengan model dan rujukan dikalibrasi ulang (Platt luar-lipatan), y_kabupaten\n" + "=" * 100)
Pk = P[(P.sasaran == "y_kabupaten") & (P.ada_positif == 1)]
for kel, sk, tags in (("biasa", "S1", B.BIASA), ("ekstrem", "S2", B.EKSTREM)):
    rows = []
    for t in tags:
        u = Pk[(Pk.skema == sk) & (Pk.tag == t)].copy(); o = O[(O.skema == sk) & (O.tag == t)]
        u["xgb_k"] = platt(o.xgb.to_numpy(), o.y.to_numpy())(u.xgb.to_numpy())
        u["ef1_k"] = platt(o.klim_frek1.to_numpy(), o.y.to_numpy())(u.klim_frek1.to_numpy())
        rows.append(u)
    g = pd.concat(rows)
    kol = ["xgb", "klim_frek1", "xgb_k", "ef1_k"]
    kab = g.gid.unique()
    blok = {t: B.Blok(x, kol, kab) for t, x in g.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.array([bs(y, b.s[m][j]) for m in kol])

    th = np.array([fn(blok[t], slice(None)) for t in tags])
    print(f"\n--- {kel}: Brier per tahun (xgb / EF-1m / xgb Platt / EF-1m Platt)")
    for t, x in zip(tags, th):
        print(f"    {t} " + " / ".join(f"{v:.5f}" for v in x))
    pas = [("xgb Platt atas EF-1m Platt", 2, 3), ("xgb mentah atas EF-1m Platt", 0, 3), ("xgb Platt atas EF-1m mentah", 2, 1),
           ("xgb mentah atas EF-1m mentah", 0, 1)]
    if kel == "biasa":
        R = B.dua_tahap(blok, fn, N_ULANG, np.random.default_rng(B.SEED), kab)
    for nama, i, r in pas:
        bt = 1 - th[:, i] / th[:, r]
        if kel == "biasa":
            cetak(f"BSS {nama}", bt.mean(), (1 - R[:, :, i] / R[:, :, r]).mean(1),
                  t_tahun(bt) + "  per tahun " + " ".join(f"{x:+.3f}" for x in bt))
        else:
            print(f"    BSS {nama:<36}" + "  ".join(f"{t} {x:+.3f}" for t, x in zip(tags, bt)))

print("\n" + "=" * 100 + "\nD3c. Kalibrasi dengan selang: xgb mentah / Platt semua / Platt pasca-2015, y_kabupaten\n" + "=" * 100)
for kel, sk, tags in (("biasa", "S1", B.BIASA), ("ekstrem", "S2", B.EKSTREM)):
    rows = []
    for t in tags:
        u = Pk[(Pk.skema == sk) & (Pk.tag == t)].copy(); o = O[(O.skema == sk) & (O.tag == t)]
        o16 = o[o.tahun_dalam >= 2016]
        u["platt_semua"] = platt(o.xgb.to_numpy(), o.y.to_numpy())(u.xgb.to_numpy())
        u["platt_pasca"] = platt(o16.xgb.to_numpy(), o16.y.to_numpy())(u.xgb.to_numpy())
        rows.append(u)
    g = pd.concat(rows)
    kol = ["xgb", "platt_semua", "platt_pasca"]
    kab = g.gid.unique()
    blok = {t: B.Blok(x, kol, kab) for t, x in g.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.r_[[b.s[m][j].mean() - y.mean() for m in kol], [bs(y, b.s[m][j]) for m in kol],
                     [slope(y, b.s[m][j]) for m in kol]]

    th = np.array([fn(blok[t], slice(None)) for t in tags])
    print(f"\n--- {kel}")
    for t, x in zip(tags, th):
        print(f"    {t} CITL " + " / ".join(f"{v:+.4f}" for v in x[:3]) + "   Brier " + " / ".join(f"{v:.5f}" for v in x[3:6])
              + "   slope " + " / ".join(f"{v:.3f}" for v in x[6:]))
    if kel == "biasa":
        R = B.dua_tahap(blok, fn, N_ULANG, np.random.default_rng(B.SEED), kab)
        for i, m in enumerate(kol):
            cetak(f"CITL {m}", th[:, i].mean(), R[:, :, i].mean(1))
            cetak(f"|CITL| {m}", np.abs(th[:, i]).mean(), np.abs(R[:, :, i]).mean(1))
            cetak(f"slope {m}", th[:, 6 + i].mean(), np.nanmean(R[:, :, 6 + i], axis=1))
        cetak("Brier pasca-2015 - Platt semua", (th[:, 5] - th[:, 4]).mean(), (R[:, :, 5] - R[:, :, 4]).mean(1))
        cetak("Brier pasca-2015 - mentah", (th[:, 5] - th[:, 3]).mean(), (R[:, :, 5] - R[:, :, 3]).mean(1))
        cetak("|CITL| pasca-2015 - Platt semua", (np.abs(th[:, 2]) - np.abs(th[:, 1])).mean(),
              (np.abs(R[:, :, 2]) - np.abs(R[:, :, 1])).mean(1))
print("\nselesai.")
