"""
Revisi fase H (Hasil_Patokan_13): kepekaan jalur MODIS 2003-2011.

Tiga varian lipatan pada panel MODIS fase G3 (DL_FIRE_MODIS/panel_modis.csv, kabupaten VIIRS, F = bulat(10 r)):
  V0 utama            latih = semua tahun MODIS lain, ambang dari 2003-2025 (sama dengan patokan_modis.py)
  V1 latih < 2016     latih = 2003-2015 tanpa tahun uji
  V2 ambang latih     latih seperti V0, ambang dan kabupaten yang dinilai dari tahun latih lipatan
Tiap varian: kalender (EF-1m / EF-3m, pemecah seri protokol utama), EF-P (enam calon fase E, Brier
tinggalkan-satu-tahun di tahun latih), EF-L (bentuk tetap), logistik riwayat lokal, GB riwayat api (fitur dan
parameter G3). V0 juga: diagnostik geser-tingkat fase E dan BSS rasio-jumlah.

Pakai (satu sasaran per jalan):
    python -u patokan_modis_h.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_h_kab.log
    python -u patokan_modis_h.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_h_nas.log
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
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
CALON = ["konstan", "konstan_bulan", "ef1", "ef3", "ef1_pasca", "ef1_halus"]
DM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_MODIS")
REF = "ef1" if SAS == "y_kabupaten" else "ef3"

# ------------------------------------------------- panel (sama dengan patokan_modis.py)
V = pd.read_csv(os.path.join(B.D, "panel_nasional_bersih.csv"), usecols=["gid", "tahun", "titik_panas"])
M = pd.read_csv(os.path.join(DM, "panel_modis.csv"), usecols=["gid", "bulan", "tahun", "bulan_ke", "titik_panas"])
M = M[M.gid.isin(V.gid.unique()) & M.tahun.between(2003, 2025)].sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
vm = V[V.tahun.between(2012, 2025)].titik_panas.sum(); mm = M[M.tahun.between(2012, 2025)].titik_panas.sum()
r = mm / vm; F = max(1, int(round(10 * r)))
GID = np.sort(M.gid.unique()); THN = np.arange(2003, 2026)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
G_, T_, M_ = M.gid.map(gi).to_numpy(), M.tahun.map(ti).to_numpy(), (M.bulan_ke - 1).to_numpy()
TP = np.full((len(GID), len(THN), 12), np.nan)
TP[G_, T_, M_] = M.titik_panas.to_numpy(float)
NB = len(THN) * 12
TPf = TP.reshape(len(GID), NB)
print(f"MODIS/VIIRS 2012-2025 r {r:.3f}; F = {F}; kabupaten {len(GID)}; sasaran {SAS}; rujukan pasangan {REF}")

# anomali nasional (tidak bergantung label)
nas = np.nansum(TPf, 0)
LNf = np.full(NB, np.nan); LNf[1:] = np.log1p(nas[:-1])
LN = LNf.reshape(len(THN), 12)


def jeda(arrf, k):
    out = np.full_like(arrf, np.nan); out[:, k:] = arrf[:, :-k]
    return out


LAG = {k: jeda(TPf, k).reshape(TP.shape) for k in (1, 2, 3, 12)}
SIN = np.broadcast_to(np.sin(2 * np.pi * (np.arange(12) + 1) / 12), TP.shape)
COS = np.broadcast_to(np.cos(2 * np.pi * (np.arange(12) + 1) / 12), TP.shape)


def label(tahun_ambang):
    """label penuh (tanpa masker) dan himpunan kabupaten dengan positif di tahun_ambang."""
    msk = np.isin(THN, tahun_ambang)
    ref = TP[:, msk, :]
    if SAS == "y_kabupaten":
        q = np.maximum(np.nanquantile(ref.reshape(len(GID), -1), 0.90, axis=1), F)
        L = (TP > q[:, None, None]).astype(float)
    else:
        L = (TP > np.nanquantile(ref, 0.90)).astype(float)
    L[np.isnan(TP)] = np.nan
    ada = np.nansum(L[:, msk, :], (1, 2)) > 0
    return L, ada


def larik(L, ada):
    Y = L.copy()
    if SAS == "y_kabupaten":
        Y[~ada] = np.nan
    Y[:, 0, 0] = np.nan
    return Y


def jendela(a):
    return a + np.roll(a, 1, axis=-1) + np.roll(a, -1, axis=-1)


def mc(tahun_latih, g, t, m, keluar_sendiri):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(TP[:, msk, :], 1)[g, m]; n = np.sum(~np.isnan(TP[:, msk, :]), 1)[g, m].astype(float)
    if keluar_sendiri:
        x = TP[g, t, m]; a = ~np.isnan(x); s = s - np.where(a, x, 0.0); n = n - a
    k = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return 1e-6 * k / (1 + k)


def rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=False):
    """calon fase E; ef1/ef3 dengan pemecah seri protokol utama (seperti patokan_modis.py)."""
    msk = np.isin(THN, tahun_latih)
    Yl = Y[:, msk, :]
    S1 = np.nansum(Yl, 1); N1 = np.sum(~np.isnan(Yl), 1)
    S3, N3 = jendela(S1), jendela(N1)
    pas = msk & (THN >= 2016)
    Sp = np.nansum(Y[:, pas, :], 1); Np = np.sum(~np.isnan(Y[:, pas, :]), 1)
    Sm = np.nansum(Yl, (0, 1)); Nm = np.sum(~np.isnan(Yl), (0, 1))
    s1, n1, s3, n3 = S1[g, m], N1[g, m].astype(float), S3[g, m], N3[g, m].astype(float)
    sp, np_ = Sp[g, m], Np[g, m].astype(float)
    sm, nm = np.full(len(g), Sm.sum()), np.full(len(g), float(Nm.sum()))
    smb, nmb = Sm[m], Nm[m].astype(float)
    if keluar_sendiri:
        yo = Y[g, t, m]; a = ~np.isnan(yo); yo0 = np.where(a, yo, 0.0)
        s1 = s1 - yo0; n1 = n1 - a
        w = np.stack([Y[g, t, (m - 1) % 12], yo, Y[g, t, (m + 1) % 12]])
        s3 = s3 - np.nansum(w, 0); n3 = n3 - np.sum(~np.isnan(w), 0)
        dp = THN[t] >= 2016
        sp = sp - np.where(dp, yo0, 0); np_ = np_ - np.where(dp, a, 0)
        sm = sm - yo0; nm = nm - a; smb = smb - yo0; nmb = nmb - a
    bagi = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), 0.0)
    pec = mc(tahun_latih, g, t, m, keluar_sendiri)
    out = {"konstan": bagi(sm, nm), "konstan_bulan": bagi(smb, nmb)}
    ef1, ef3 = bagi(s1, n1), bagi(s3, n3)
    out["ef1"], out["ef3"] = ef1 + pec, ef3 + pec
    out["ef1_pasca"] = np.where(np_ > 0, bagi(sp, np_), out["konstan_bulan"])
    out["ef1_halus"] = (s1 + KAPPA * ef3) / (n1 + KAPPA)
    return out


def anomali(tahun_latih, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(LN[msk], 0); n = np.sum(~np.isnan(LN[msk]), 0).astype(float)
    s, n = s[m], n[m]
    if keluar_sendiri:
        a = ~np.isnan(LN[t, m]); s = s - np.where(a, LN[t, m], 0.0); n = n - a
    return LN[t, m] - s / np.maximum(n, 1)


def logit(p, e=1e-4):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def indeks(Y, tahun):
    g, t, m = np.where(~np.isnan(Y[:, np.isin(THN, tahun), :]))
    t = np.flatnonzero(np.isin(THN, tahun))[t]
    return g, t, m


def X_lokal(g, t, m, ef1, YL1):
    return np.column_stack([logit(ef1), np.nan_to_num(YL1[g, t, m])]
                           + [np.log1p(np.nan_to_num(LAG[k][g, t, m])) for k in (1, 2, 3)]
                           + [SIN[g, t, m], COS[g, t, m]])


def X_gb(g, t, m, ef1, a, YL1):
    return np.column_stack([np.log1p(np.nan_to_num(LAG[k][g, t, m])) for k in (1, 2, 3, 12)]
                           + [SIN[g, t, m], COS[g, t, m], ef1, np.nan_to_num(YL1[g, t, m]), a])


def lipatan(u, varian):
    tl = [x for x in THN if x != u and (varian != "V1" or x < 2016)]
    L, ada = label(THN if varian != "V2" else tl)
    Y = larik(L, ada)
    YL1 = jeda(L.reshape(len(GID), NB), 1).reshape(L.shape)   # keadaan bulan lalu dari label lipatan ini
    br = {c: [] for c in CALON}
    for v in tl:
        g, t, m = indeks(Y, [v])
        rr = rujukan(Y, [x for x in tl if x != v], g, t, m)
        for c in CALON:
            br[c].append(np.mean((rr[c] - Y[g, t, m]) ** 2))
    pilih = min(CALON, key=lambda c: np.mean(br[c]))
    g, t, m = indeks(Y, tl)
    rl = rujukan(Y, tl, g, t, m, keluar_sendiri=True)
    an = anomali(tl, t, m, keluar_sendiri=True)
    yl = Y[g, t, m]
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack([logit(rl["ef1_halus"]), an]), yl)
    lok = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_lokal(g, t, m, rl["ef1"], YL1), yl)
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(X_gb(g, t, m, rl["ef1"], an, YL1), yl)
    g2, t2, m2 = indeks(Y, [u])
    ru = rujukan(Y, tl, g2, t2, m2); au = anomali(tl, t2, m2)
    b = pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2], "kalender": ru[REF], "efp": ru[pilih],
                      "efl": lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), au]))[:, 1],
                      "lokal": lok.predict_proba(X_lokal(g2, t2, m2, ru["ef1"], YL1))[:, 1],
                      "gb": xg.predict_proba(X_gb(g2, t2, m2, ru["ef1"], au, YL1))[:, 1]})
    print(f"  {varian} {u}: latih {len(tl)} tahun ({tl[0]}-{tl[-1]}), {len(yl):,} baris; uji {len(b):,}, "
          f"prevalensi {b.y.mean() * 100:.2f}%; EF-P {pilih}; koef EF-L {lg.coef_[0][0]:+.3f} {lg.coef_[0][1]:+.3f}",
          flush=True)
    return b


def geser(f, ybar):
    lf = logit(f, 1e-6)
    h = lambda d: np.mean(1 / (1 + np.exp(-(lf + d)))) - ybar
    return 1 / (1 + np.exp(-(lf + brentq(h, -30, 30))))


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def cetak(nama, pt, dist):
    lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
    lolos = lo > 0 and tl > 0
    print(f"    {nama:<30}{pt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
          f"{'  LOLOS' if lolos else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
    return lolos


B.cek_ap()
tags = [str(u) for u in UJI]
for varian in ("V0", "V1", "V2"):
    print("\n" + "=" * 100 + f"\nH {varian}  {SAS}  (rujukan pasangan {REF}), n = 9, df 8\n" + "=" * 100, flush=True)
    P = pd.concat([lipatan(u, varian) for u in UJI], ignore_index=True)
    KOL = ["kalender", "efp", "efl", "lokal", "gb"]
    if varian == "V0":
        for c in ("kalender", "efp", "gb"):
            P[c + "_lv"] = np.nan
            for _, x in P.groupby("tag"):
                P.loc[x.index, c + "_lv"] = geser(x[c].to_numpy(), x.y.mean())
        KOL += ["kalender_lv", "efp_lv", "gb_lv"]
    ix = {c: i for i, c in enumerate(KOL)}; nk = len(KOL)
    kab = P.gid.unique()
    blok = {t: B.Blok(x, KOL, kab) for t, x in P.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.array([np.mean((b.s[c][j] - y) ** 2) for c in KOL] + [B.ap_cepat(y, b.s[c][j]) for c in KOL])

    th = np.array([fn(blok[t], slice(None)) for t in tags])
    R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)
    print("  Brier rata-rata: " + "  ".join(f"{c} {th[:, ix[c]].mean():.5f}" for c in KOL[:5]))
    print("  AUC-PR rata-rata: " + "  ".join(f"{c} {th[:, nk + ix[c]].mean():.3f}" for c in KOL[:5]))
    print("  BSS")
    lolos = {}
    for a, rr in (("efl", "kalender"), ("efp", "kalender"), ("efl", "efp"), ("lokal", "efl"),
                  ("gb", "kalender"), ("gb", "efp"), ("gb", "efl")):
        i, k = ix[a], ix[rr]
        lolos[(a, rr)] = cetak(f"{a} atas {rr}", 1 - th[:, i] / th[:, k], (1 - R2[:, :, i] / R2[:, :, k]).mean(1))
    print("  AUC-PR")
    for a, rr in (("gb", "efl"), ("gb", "kalender")):
        i, k = nk + ix[a], nk + ix[rr]
        cetak(f"{a} - {rr}", th[:, i] - th[:, k], (R2[:, :, i] - R2[:, :, k]).mean(1))
    semua = all(lolos[("gb", x)] for x in ("kalender", "efp", "efl"))
    print(f"  R36c {varian}: GB lolos atas kalender {lolos[('gb', 'kalender')]}, EF-P {lolos[('gb', 'efp')]}, "
          f"EF-L {lolos[('gb', 'efl')]}  -> mengalahkan ketiga rujukan: {semua}", flush=True)
    if varian == "V0":
        print("  Geser-tingkat (BSS mentah / tingkat-benar / bagian yang hilang)")
        for acu in ("kalender", "efp"):
            i, k, il, kl = ix["gb"], ix[acu], ix["gb_lv"], ix[acu + "_lv"]
            mentah = 1 - th[:, i] / th[:, k]; lv = 1 - th[:, il] / th[:, kl]
            a_ = (1 - R2[:, :, i] / R2[:, :, k]).mean(1); b_ = (1 - R2[:, :, il] / R2[:, :, kl]).mean(1)
            bag = 1 - b_ / a_
            lo, hi = np.nanpercentile(b_, [2.5, 97.5]); blo, bhi = np.nanpercentile(bag, [2.5, 97.5]); tl, tu = t95(lv)
            print(f"    gb atas {acu:<9} mentah {mentah.mean():+.4f}  tingkat-benar {lv.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] "
                  f"t95 [{tl:+.4f}, {tu:+.4f}]  bagian hilang {1 - lv.mean() / mentah.mean():.2f} [{blo:.2f}, {bhi:.2f}]", flush=True)
        print("  BSS rasio-jumlah (semua baris uji) vs rata-rata rasio")
        for a, rr in (("efl", "kalender"), ("gb", "kalender"), ("gb", "efp"), ("gb", "efl"), ("lokal", "efl")):
            rs = 1 - np.mean((P[a] - P.y) ** 2) / np.mean((P[rr] - P.y) ** 2)
            mr = (1 - th[:, ix[a]] / th[:, ix[rr]]).mean()
            print(f"    {a} atas {rr:<9} rasio-jumlah {rs:+.4f}  rata-rata rasio {mr:+.4f}"
                  f"{'  TANDA BERBALIK' if np.sign(rs) != np.sign(mr) else ''}", flush=True)
print("selesai.")
