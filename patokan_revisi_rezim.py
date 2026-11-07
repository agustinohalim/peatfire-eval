"""
PeatFireBench revisi fase D8 (Hasil_Patokan_9): apakah kebakaran sesudah 2015 lebih rendah per satuan
kekeringan? Deskriptif, bukan prakiraan. Kabupaten gambut (gambut_fraksi >= 0,10; kepekaan > 0),
Agustus-Oktober 2013-2025. Regresi logistik y_kabupaten dan regresi linear log1p titik panas pada log DC
bulan yang sama (GFWED MERRA2.CORRECTED), efek tetap bulan, dan indikator pasca-2015 (tahun >= 2016).
Selang koefisien pasca-2015 dari bootstrap gugus tahun (1.000, seed 2026); ulangan tanpa tahun
sebelum 2016 atau tanpa tahun sesudahnya dibuang dan banyaknya dicetak.

Pakai:    python -u patokan_revisi_rezim.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_rezim.log
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B
from kelompok_fitur import PANEL

df = pd.read_csv(f"{B.D}/{PANEL}", usecols=["gid", "provinsi", "tahun", "bulan", "bulan_ke", "titik_panas", "y_kabupaten"])
C = pd.read_csv(f"{B.D}/gfwed_komponen_MERRA2.CORRECTED.csv", usecols=["GID_2", "bulan", "dc"]).rename(columns={"GID_2": "gid"})
G = pd.read_csv(f"{B.D}/gambut_kabupaten.csv")
df = df.merge(C, on=["gid", "bulan"], how="left", validate="one_to_one")
df["gambut"] = df.gid.map(G.set_index("gid").gambut_fraksi).fillna(0.0)
df = df[df.bulan_ke.isin([8, 9, 10]) & df.tahun.between(2013, 2025)].copy()
df["pasca"] = (df.tahun >= 2016).astype(float)
df["ldc"] = np.log(df.dc.clip(lower=1.0))


def X(d):
    return np.column_stack([d.ldc, d.pasca, (d.bulan_ke == 9).astype(float), (d.bulan_ke == 10).astype(float)])


def koef(d):
    x = X(d)
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(x, d.y_kabupaten)
    xl = np.column_stack([np.ones(len(d)), x])
    beta = np.linalg.lstsq(xl, np.log1p(d.titik_panas.to_numpy()), rcond=None)[0]
    return lg.coef_[0, 0], lg.coef_[0, 1], beta[1], beta[2]


for nama, f in (("gambut >= 0,10", lambda g: g >= 0.10), ("bergambut > 0", lambda g: g > 0)):
    d = df[f(df.gambut)]
    assert d.dc.notna().all()
    print("=" * 100 + f"\nD8. {nama}: {d.gid.nunique()} kabupaten, {len(d)} kabupaten-bulan Agu-Okt 2013-2025\n" + "=" * 100)
    th = d.groupby("tahun").agg(dc=("dc", "mean"), y=("y_kabupaten", "mean"), tp=("titik_panas", "sum"))
    k0 = koef(d)
    # residu per tahun dari model tanpa indikator pasca (logistik): rata-rata y teramati - rata-rata p
    x0 = X(d)[:, [0, 2, 3]]
    m0 = LogisticRegression(C=1e6, max_iter=2000).fit(x0, d.y_kabupaten)
    d = d.assign(p0=m0.predict_proba(x0)[:, 1])
    th["p_dari_dc"] = d.groupby("tahun").p0.mean()
    th["y_minus_p"] = th.y - th.p_dari_dc
    print(th.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n  logistik: log DC {k0[0]:+.3f}, pasca-2015 {k0[1]:+.3f} (rasio odds {np.exp(k0[1]):.2f})")
    print(f"  linear log1p titik panas: log DC {k0[2]:+.3f}, pasca-2015 {k0[3]:+.3f}")
    rng = np.random.default_rng(B.SEED)
    tahun = np.array(sorted(d.tahun.unique()))
    per = {t: g for t, g in d.groupby("tahun")}
    bs, buang = [], 0
    for _ in range(1000):
        tt = rng.choice(tahun, len(tahun))
        if (tt < 2016).sum() == 0 or (tt >= 2016).sum() == 0:
            buang += 1; continue
        bs.append(koef(pd.concat([per[t] for t in tt])))
    bs = np.array(bs)
    for i, n in ((1, "logistik pasca-2015"), (3, "linear pasca-2015"), (0, "logistik log DC")):
        lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
        print(f"  {n:<24}{k0[i]:+.3f} [{lo:+.3f}, {hi:+.3f}]  {'di bawah nol' if hi < 0 else ''}")
    print(f"  ulangan dibuang (tanpa tahun pra- atau pasca-2015): {buang}; tahun pra-2016 dalam data: {(tahun < 2016).sum()}\n")
print("selesai.")
