"""
PeatFireBench revisi fase G3 (Hasil_Patokan_12): EF-L pada tahun MODIS 2003-2011.

Panel DL_FIRE_MODIS/panel_modis.csv (modis_keyakinan.py + bangun_panel_nasional.js tanpa perubahan),
kabupaten yang sama dengan panel VIIRS. Sasaran dengan aturan panel VIIRS dari MODIS 2003-2025; batas bawah
per kabupaten F = bulat(10 x r), r = MODIS nasional / VIIRS nasional 2012-2025, minimal 1.
Lipatan: tiap tahun uji u = 2003..2011 ditahan, semua tahun MODIS lain tahun latih (seperti S2).
EF-L dengan bentuk fase E/F (kappa 2, logit ef1_halus + anomali log1p jumlah nasional bulan lalu, baris latih
tinggalkan-tahun-sendiri); hanya tiga koefisiennya dilatih. Pembanding: konstan, EF-1m, EF-3m, ef1_halus,
logistik riwayat lokal (bentuk log_ef1_jeda), GB riwayat api (parameter baku; log1p jeda 1, 2, 3, 12,
sin/cos bulan, EF-1m, keadaan lampau bulan lalu, anomali nasional). Januari 2003 dibuang (tanpa jeda).
EF-1m dan EF-3m memakai pemecah seri protokol utama (+1e-6 k/(1+k), k = rata-rata jumlah tahun latih,
R.klim_frek); jeda kosong diisi 0 seperti xgb_sejarah.

Pakai (satu sasaran per jalan):
    python -u patokan_modis.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_kab.log
    python -u patokan_modis.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_nas.log
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
KAPPA = 2.0
N = 1000
UJI = list(range(2003, 2012))
DM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_MODIS")

# ------------------------------------------------- panel dan sasaran
V = pd.read_csv(os.path.join(B.D, "panel_nasional_bersih.csv"), usecols=["gid", "tahun", "titik_panas"])
M = pd.read_csv(os.path.join(DM, "panel_modis.csv"), usecols=["gid", "bulan", "tahun", "bulan_ke", "titik_panas"])
M = M[M.gid.isin(V.gid.unique()) & M.tahun.between(2003, 2025)].sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
vm = V[V.tahun.between(2012, 2025)].titik_panas.sum(); mm = M[M.tahun.between(2012, 2025)].titik_panas.sum()
r = mm / vm; F = max(1, int(round(10 * r)))
q_kab = np.maximum(M.groupby("gid").titik_panas.quantile(0.90), F)
gab = M.titik_panas.quantile(0.90)
M["y_kabupaten"] = (M.titik_panas > M.gid.map(q_kab)).astype(int)
M["y_gabungan"] = (M.titik_panas > gab).astype(int)
ADA_G = M.groupby("gid").y_kabupaten.max()
print(f"MODIS 2012-2025 {mm:,} / VIIRS {vm:,} = r {r:.3f}; batas bawah F = {F}; ambang gabungan {gab:.1f}")
print(f"kabupaten {M.gid.nunique()}, dengan positif per kabupaten {int(ADA_G.sum())}; "
      f"prevalensi {M.y_kabupaten.mean() * 100:.2f}% / {M.y_gabungan.mean() * 100:.2f}%")
print("deteksi MODIS per tahun: " + " ".join(f"{t}:{v // 1000}k" for t, v in M.groupby("tahun").titik_panas.sum().items()))

g_ = M.groupby("gid")
for k in (1, 2, 3, 12):
    M[f"tp_lag{k}"] = g_.titik_panas.shift(k)
M["y_lag1"] = g_[SAS].shift(1)
M["bulan_sin"], M["bulan_cos"] = np.sin(2 * np.pi * M.bulan_ke / 12), np.cos(2 * np.pi * M.bulan_ke / 12)

GID = np.sort(M.gid.unique()); THN = np.arange(2003, 2026)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
M["_g"], M["_t"], M["_m"] = M.gid.map(gi), M.tahun.map(ti), M.bulan_ke - 1
nas = M.groupby(["tahun", "bulan_ke"]).titik_panas.sum().sort_index()
lag = nas.shift(1)
LN = np.full((len(THN), 12), np.nan)
for (t, b), v in lag.items():
    if not np.isnan(v):
        LN[ti[t], b - 1] = np.log1p(v)
Y = np.full((len(GID), len(THN), 12), np.nan)
Y[M._g, M._t, M._m] = M[SAS].to_numpy(float)
if SAS == "y_kabupaten":
    Y[~np.isin(GID, ADA_G[ADA_G > 0].index)] = np.nan
Y[:, 0, 0] = np.nan  # Januari 2003: tanpa jeda
TP = np.full((len(GID), len(THN), 12), np.nan)
TP[M._g, M._t, M._m] = M.titik_panas.to_numpy(float)


def mc(tahun_latih, g, t, m, keluar_sendiri=False):
    """rata-rata jumlah tahun latih untuk pemecah seri (R.klim_frek), tanpa tahun baris itu bila diminta."""
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(TP[:, msk, :], 1)[g, m]; n = np.sum(~np.isnan(TP[:, msk, :]), 1)[g, m].astype(float)
    if keluar_sendiri:
        x = TP[g, t, m]; ada = ~np.isnan(x)
        s = s - np.where(ada, x, 0.0); n = n - ada
    k = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return 1e-6 * k / (1 + k)


# ------------------------------------------------- EF-L (bentuk fase E/F)
def jendela(a):
    return a + np.roll(a, 1, axis=-1) + np.roll(a, -1, axis=-1)


def rujukan(tahun_latih, g, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    Yl = Y[:, msk, :]
    S1 = np.nansum(Yl, 1); N1 = np.sum(~np.isnan(Yl), 1)
    S3, N3 = jendela(S1), jendela(N1)
    s1, n1, s3, n3 = S1[g, m], N1[g, m].astype(float), S3[g, m], N3[g, m].astype(float)
    if keluar_sendiri:
        yo = Y[g, t, m]; ada = ~np.isnan(yo); yo0 = np.where(ada, yo, 0.0)
        s1 = s1 - yo0; n1 = n1 - ada
        w = np.stack([Y[g, t, (m - 1) % 12], yo, Y[g, t, (m + 1) % 12]])
        s3 = s3 - np.nansum(w, 0); n3 = n3 - np.sum(~np.isnan(w), 0)
    bagi = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), 0.0)
    ef1, ef3 = bagi(s1, n1), bagi(s3, n3)
    pec = mc(tahun_latih, g, t, m, keluar_sendiri)
    return {"ef1": ef1 + pec, "ef3": ef3 + pec, "ef1_halus": (s1 + KAPPA * ef3) / (n1 + KAPPA)}


def anomali(tahun_latih, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(LN[msk], 0); n = np.sum(~np.isnan(LN[msk]), 0).astype(float)
    s, n = s[m], n[m]
    if keluar_sendiri:
        ada = ~np.isnan(LN[t, m])
        s = s - np.where(ada, LN[t, m], 0.0); n = n - ada
    return LN[t, m] - s / np.maximum(n, 1)


def logit(p, e=1e-4):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def indeks(tahun):
    g, t, m = np.where(~np.isnan(Y[:, np.isin(THN, tahun), :]))
    t = np.flatnonzero(np.isin(THN, tahun))[t]
    return g, t, m


KOL = ["konstan", "ef1", "ef3", "ef1_halus", "ef1_tingkat", "lokal", "gb"]
lookup = M.set_index(["_g", "_t", "_m"])
baris = []
for u in UJI:
    tl = [x for x in THN if x != u]
    g, t, m = indeks(tl)
    rl = rujukan(tl, g, t, m, keluar_sendiri=True)
    an = anomali(tl, t, m, keluar_sendiri=True)
    yl = Y[g, t, m]
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack([logit(rl["ef1_halus"]), an]), yl)
    L = lookup.loc[list(zip(g, t, m))]
    g2, t2, m2 = indeks([u])
    ru = rujukan(tl, g2, t2, m2); au = anomali(tl, t2, m2)
    U = lookup.loc[list(zip(g2, t2, m2))]

    def X_lokal(d, ef1):
        return np.column_stack([logit(ef1), d.y_lag1.fillna(0).to_numpy()]
                               + [np.log1p(d[f"tp_lag{k}"].fillna(0).to_numpy()) for k in (1, 2, 3)]
                               + [d.bulan_sin.to_numpy(), d.bulan_cos.to_numpy()])

    def X_gb(d, ef1, a):
        return np.column_stack([np.log1p(d[f"tp_lag{k}"].fillna(0).to_numpy()) for k in (1, 2, 3, 12)]
                               + [d.bulan_sin.to_numpy(), d.bulan_cos.to_numpy(), ef1, d.y_lag1.fillna(0).to_numpy(), a])

    lok = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_lokal(L, rl["ef1"]), yl)
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(X_gb(L, rl["ef1"], an), yl)
    b = pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2], "konstan": yl.mean(),
                      "ef1": ru["ef1"], "ef3": ru["ef3"], "ef1_halus": ru["ef1_halus"],
                      "ef1_tingkat": lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), au]))[:, 1],
                      "lokal": lok.predict_proba(X_lokal(U, ru["ef1"]))[:, 1],
                      "gb": xg.predict_proba(X_gb(U, ru["ef1"], au))[:, 1]})
    baris.append(b)
    print(f"{u}: latih {len(yl):,} baris, uji {len(b):,}, prevalensi uji {b.y.mean() * 100:.2f}%; "
          f"koef EF-L {lg.coef_[0][0]:+.3f} {lg.coef_[0][1]:+.3f}", flush=True)
P = pd.concat(baris, ignore_index=True)


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


kab = P.gid.unique(); ix = {c: i for i, c in enumerate(KOL)}; nk = len(KOL)
blok = {t: B.Blok(x, KOL, kab) for t, x in P.groupby("tag")}
tags = [str(u) for u in UJI]


def fn(b, j):
    y = b.y[j]
    return np.array([np.mean((b.s[c][j] - y) ** 2) for c in KOL] + [B.ap_cepat(y, b.s[c][j]) for c in KOL])


th = np.array([fn(blok[t], slice(None)) for t in tags])
R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)
REF = "ef1" if SAS == "y_kabupaten" else "ef3"
print("\n" + "=" * 100 + f"\nG3. {SAS}: tahun MODIS 2003-2011 (rujukan pasangan {REF}), n = 9, df 8\n" + "=" * 100)
print("  Brier rata-rata: " + "  ".join(f"{c} {th[:, ix[c]].mean():.5f}" for c in KOL))
print("  AUC-PR rata-rata: " + "  ".join(f"{c} {th[:, nk + ix[c]].mean():.3f}" for c in KOL))


def cetak(nama, pt, dist):
    lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
    print(f"    {nama:<34}{pt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
          f"{'  LOLOS' if lo > 0 and tl > 0 else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)


print("  BSS")
for a, rr in (("ef1_tingkat", REF), ("konstan", REF), ("ef1_halus", REF), ("lokal", REF), ("gb", REF),
              ("lokal", "ef1_tingkat"), ("gb", "ef1_tingkat")):
    i, k = ix[a], ix[rr]
    cetak(f"{a} atas {rr}", 1 - th[:, i] / th[:, k], (1 - R2[:, :, i] / R2[:, :, k]).mean(1))
print("  AUC-PR")
for a, rr in (("ef1_tingkat", REF), ("lokal", REF), ("gb", REF), ("lokal", "ef1_tingkat"), ("gb", "ef1_tingkat")):
    i, k = nk + ix[a], nk + ix[rr]
    cetak(f"{a} - {rr}", th[:, i] - th[:, k], (R2[:, :, i] - R2[:, :, k]).mean(1))
print("  per tahun, selang kabupaten (BSS EF-L atas rujukan; BSS GB atas EF-L)")
for t in tags:
    D_ = B.kab_tetap(blok[t], fn, N, np.random.default_rng(B.SEED), kab)
    pt = th[tags.index(t)]
    a1 = 1 - D_[:, ix["ef1_tingkat"]] / D_[:, ix[REF]]; a2 = 1 - D_[:, ix["gb"]] / D_[:, ix["ef1_tingkat"]]
    print(f"    {t}: prev {blok[t].y.mean() * 100:5.2f}%  EF-L/{REF} {1 - pt[ix['ef1_tingkat']] / pt[ix[REF]]:+.3f} "
          f"[{np.nanpercentile(a1, 2.5):+.3f}, {np.nanpercentile(a1, 97.5):+.3f}]  GB/EF-L "
          f"{1 - pt[ix['gb']] / pt[ix['ef1_tingkat']]:+.3f} [{np.nanpercentile(a2, 2.5):+.3f}, {np.nanpercentile(a2, 97.5):+.3f}]", flush=True)
print("selesai.")
