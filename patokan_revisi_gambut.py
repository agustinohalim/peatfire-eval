"""
PeatFireBench revisi fase D6 (Hasil_Patokan_9): Tabel 2 per strata gambut (gambut_fraksi > 0, = 0,
>= 0,10) dan keunggulan xgb atas EF-1m per provinsi.

Pakai:    python -u patokan_revisi_gambut.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_gambut.log
"""
import numpy as np
import pandas as pd
from scipy import stats

import patokan_revisi_bantu as B

MODEL = ["rf", "xgb", "mlp", "log_ef1_jeda"]
B.cek_ap()
P = B.muat_P(fwi=False)
S = pd.read_csv(f"{B.D}/patokan_revisi_sederhana.csv", dtype={"tag": str})
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = P.merge(S[kunci + ["log_ef1_jeda"]], on=kunci, how="left", validate="one_to_one")
G = pd.read_csv(f"{B.D}/gambut_kabupaten.csv")
P["gambut"] = P.gid.map(G.set_index("gid").gambut_fraksi).fillna(0.0)
STRATA = {"bergambut (>0)": lambda x: x > 0, "tak bergambut (=0)": lambda x: x == 0, "gambut >= 0,10": lambda x: x >= 0.10}


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


print("=" * 100 + "\nD6. Strata gambut\n" + "=" * 100)
for sas, ref in B.PASANGAN.items():
    d = B.ambil_sasaran(P, sas)
    kol = MODEL + [ref]
    for nama, f in STRATA.items():
        ds = d[f(d.gambut)]
        s1 = ds[(ds.skema == "S1") & ds.tag.isin(B.BIASA)]
        kab = s1.gid.unique()
        blok = {t: B.Blok(g, kol, kab) for t, g in s1.groupby("tag")}

        def fn(b, j):
            y = b.y[j]
            return np.array([B.ap_cepat(y, b.s[m][j]) for m in kol])

        th = np.array([fn(blok[t], slice(None)) for t in B.BIASA])
        R = B.dua_tahap(blok, fn, 1000, np.random.default_rng(B.SEED), kab)
        print(f"\n--- {sas}, {nama}: {len(kab)} kabupaten, prevalensi {s1.y.mean() * 100:.2f}%, "
              f"AP {B.LABEL[ref]} {np.nanmean(th[:, -1]):.3f}; positif per tahun "
              + " ".join(f"{t} {int(blok[t].y.sum())}" for t in B.BIASA))
        for i, m in enumerate(MODEL):
            pt = th[:, i] - th[:, -1]
            dist = np.nanmean(R[:, :, i] - R[:, :, -1], axis=1)
            lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, thh = t95(pt[~np.isnan(pt)])
            print(f"    {m:<14}{np.nanmean(pt):+.3f}  2tahap [{lo:+.3f}, {hi:+.3f}] (lebar {hi - lo:.3f})  t95 [{tl:+.3f}, {thh:+.3f}]"
                  f"  {'LOLOS R16' if lo > 0 and tl > 0 else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
        s2 = ds[ds.skema == "S2"]
        for t in B.EKSTREM:
            g = s2[s2.tag == t]
            if g.y.sum() == 0:
                continue
            kab2 = g.gid.unique(); b = B.Blok(g, kol, kab2)
            r0 = fn(b, slice(None))
            kb = B.kab_tetap(b, fn, 500, np.random.default_rng(B.SEED), kab2)
            print(f"    S2 {t} ({int(g.y.sum())} positif, AP {B.LABEL[ref]} {r0[-1]:.3f}): " + "  ".join(
                f"{m} {r0[i] - r0[-1]:+.3f} [{np.nanpercentile(kb[:, i] - kb[:, -1], 2.5):+.3f}, "
                f"{np.nanpercentile(kb[:, i] - kb[:, -1], 97.5):+.3f}]" for i, m in enumerate(MODEL[:2])))

print("\n" + "=" * 100 + "\nD6. Provinsi: keunggulan xgb dan rf atas EF-1m, per kabupaten, rata-rata tahun biasa\n" + "=" * 100)
d = B.ambil_sasaran(P, "y_kabupaten")
s1 = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
baris = []
for prov, g in s1.groupby("provinsi"):
    nkab = g.gid.nunique()
    if nkab < 5:
        continue
    v = {m: [] for m in ("xgb", "rf", "klim_frek1")}
    for t, x in g.groupby("tag"):
        if x.y.sum() == 0:
            continue
        for m in v:
            v[m].append(B.ap_cepat(x.y, x[m]))
    if not v["xgb"]:
        continue
    baris.append((prov, nkab, int((g.drop_duplicates("gid").gambut > 0).sum()), len(v["xgb"]), np.mean(v["klim_frek1"]),
                  np.mean(v["xgb"]) - np.mean(v["klim_frek1"]), np.mean(v["rf"]) - np.mean(v["klim_frek1"]), int(g.y.sum())))
T = pd.DataFrame(baris, columns=["provinsi", "kab", "kab_gambut", "tahun", "AP_EF1m", "xgb-EF1m", "rf-EF1m", "positif"])
T = T.sort_values("xgb-EF1m", ascending=False)
print(T.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print(f"\nprovinsi dengan xgb > EF-1m: {(T['xgb-EF1m'] > 0).sum()} dari {len(T)}; rf > EF-1m: {(T['rf-EF1m'] > 0).sum()}")
print("\nselesai.")
