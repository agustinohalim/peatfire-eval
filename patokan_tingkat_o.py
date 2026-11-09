"""
Revisi fase O (Hasil_Patokan_20), panel VIIRS, tahun biasa 2020-2025 (skema S1), satu sasaran per jalan.
  O1  xgb_nasdc = GB baku seperti xgb_nas fase F (patokan_revisi_tingkat.py: R.FITUR, 300 pohon, depth 3, lr 0,05,
      subsample/colsample 0,9, reg_lambda 1, benih 42) + anomali_nasional (fase F) + anomali_dc (fase M), jadi
      melihat semua masukan EF-LD. Latih = tahun sebelum tahun uji. EF-LD dihitung ulang seperti fase M.
      Sel: xgb_nasdc atas EF-LD, EF-L, kalender pasangan, EF-P; atas xgb_nas dan xgb (deskriptif);
      reproduksi xgb atas EF-LD (fase M). Batas bawah dua tahap lima benih + 10.000 ulangan untuk xgb_nasdc atas
      EF-LD dan atas EF-L.
Fungsi panel, EF-L, anomali dan anomali DC disalin dari patokan_tingkat_m.py tanpa perubahan isi.

Pakai:
    python -u patokan_tingkat_o.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_o_kab.log
    python -u patokan_tingkat_o.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_o_nas.log
Keluaran: DL_FIRE_NASIONAL/patokan_tingkat_o_gb_<sasaran>.csv (tidak masuk repo)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
REF = B.PASANGAN[SAS]
KAPPA = 2.0
KUNCI = ["sasaran", "skema", "tag", "gid", "bulan"]
UJI = list(range(2020, 2026))

# ---------------------------------------------------------------- panel sebagai larik (sama dengan fase E)
df = pd.read_csv(os.path.join(B.D, "panel_nasional_fitur_penuh.csv"),
                 usecols=["gid", "tahun", "bulan", "bulan_ke", "tp_lag1", "y_kabupaten", "y_gabungan", "ada_positif"])
GID = np.sort(df.gid.unique()); THN = np.arange(df.tahun.min(), df.tahun.max() + 1)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
df["_g"] = df.gid.map(gi); df["_t"] = df.tahun.map(ti); df["_m"] = df.bulan_ke - 1
ADA = df.groupby("_g").ada_positif.max().reindex(range(len(GID))).fillna(0).to_numpy() == 1
NAS = df.groupby(["_t", "_m"]).tp_lag1.sum()
LN = np.full((len(THN), 12), np.nan)
LN[NAS.index.get_level_values(0), NAS.index.get_level_values(1)] = np.log1p(NAS.to_numpy())

C = pd.read_csv(os.path.join(B.D, "gfwed_komponen_MERRA2.CORRECTED.csv"), usecols=["GID_2", "bulan", "dc"])
C = C.rename(columns={"GID_2": "gid"}).sort_values(["gid", "bulan"])
C["dc_lag1"] = C.groupby("gid").dc.shift(1)
x = df[["gid", "bulan", "_g", "_t", "_m"]].merge(C[["gid", "bulan", "dc_lag1"]], on=["gid", "bulan"], how="left",
                                                 validate="one_to_one")
print(f"DC bulan lalu hilang {x.dc_lag1.isna().mean() * 100:.2f}% (diisi median kabupaten)", flush=True)
x["dc_lag1"] = x.dc_lag1.fillna(x.groupby("gid").dc_lag1.transform("median")).fillna(x.dc_lag1.median())
DC = np.full((len(GID), len(THN), 12), np.nan)
DC[x._g, x._t, x._m] = np.log1p(x.dc_lag1.to_numpy())
del x, C


def larik(ylab):
    Y = np.full((len(GID), len(THN), 12), np.nan)
    Y[df._g, df._t, df._m] = df[ylab].to_numpy(float)
    if ylab == "y_kabupaten":
        Y[~ADA] = np.nan
    return Y


def jendela(a):
    return a + np.roll(a, 1, axis=-1) + np.roll(a, -1, axis=-1)


def ef1_halus(Y, tahun_latih, g, t, m, keluar_sendiri=False):
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
    ef3 = np.where(n3 > 0, s3 / np.maximum(n3, 1), 0.0)
    return (s1 + KAPPA * ef3) / (n1 + KAPPA)


def anomali(tahun_latih, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(LN[msk], 0); n = np.sum(~np.isnan(LN[msk]), 0).astype(float)
    s, n = s[m], n[m]
    if keluar_sendiri:
        s = s - LN[t, m]; n = n - 1
    return LN[t, m] - s / np.maximum(n, 1)


def anomali_dc(tahun_latih, g, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(DC[:, msk, :], 1)[g, m]; n = np.sum(~np.isnan(DC[:, msk, :]), 1)[g, m].astype(float)
    if keluar_sendiri:
        s = s - DC[g, t, m]; n = n - 1
    return DC[g, t, m] - s / np.maximum(n, 1)


def logit(p, e=1e-4):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def indeks(Y, tahun):
    g, t, m = np.where(~np.isnan(Y[:, np.isin(THN, tahun), :]))
    t = np.flatnonzero(np.isin(THN, tahun))[t]
    return g, t, m


Y = larik(SAS)
bag = []
print("\nM1 koefisien per lipatan (EF-L ulang | EF-D | EF-LD)")
for u in UJI:
    tl = [v for v in THN if v < u]
    g, t, m = indeks(Y, tl)
    h, a, d = logit(ef1_halus(Y, tl, g, t, m, True)), anomali(tl, t, m, True), anomali_dc(tl, g, t, m, True)
    yl = Y[g, t, m]
    fit = lambda *X: LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack(X), yl)
    mL, mD, mLD = fit(h, a), fit(h, d), fit(h, a, d)
    g, t, m = indeks(Y, [u])
    h, a, d = logit(ef1_halus(Y, tl, g, t, m)), anomali(tl, t, m), anomali_dc(tl, g, t, m)
    bag.append(pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[p]}-{q + 1:02d}" for p, q in zip(t, m)], "tag": str(u),
                             "efl_ulang": mL.predict_proba(np.column_stack([h, a]))[:, 1],
                             "efd": mD.predict_proba(np.column_stack([h, d]))[:, 1],
                             "efld": mLD.predict_proba(np.column_stack([h, a, d]))[:, 1]}))
    print(f"  {u}: EF-L {mL.coef_[0, 0]:+.3f} {mL.coef_[0, 1]:+.3f} | EF-D {mD.coef_[0, 0]:+.3f} DC {mD.coef_[0, 1]:+.3f} | "
          f"EF-LD {mLD.coef_[0, 0]:+.3f} nas {mLD.coef_[0, 1]:+.3f} DC {mLD.coef_[0, 2]:+.3f}", flush=True)
BARU = pd.concat(bag, ignore_index=True)
BARU["sasaran"], BARU["skema"] = SAS, "S1"
del df, Y

# ---------------------------------------------------------------- O1: GB dengan anomali nasional dan DC
OUTGB = os.path.join(B.D, f"patokan_tingkat_o_gb_{SAS}.csv")
if not os.path.exists(OUTGB):
    Dp = R.df.copy()
    Dp["_g"] = Dp.gid.map(gi); Dp["_t"] = Dp.tahun.map(ti); Dp["_m"] = Dp.bulan_ke - 1
    assert not Dp[["_g", "_t"]].isna().any().any()
    FIT = R.FITUR + ["anomali_nasional", "anomali_dc"]
    baris = []
    for u in UJI:
        tl = [v for v in THN if v < u]
        latih = Dp[Dp.tahun.isin(tl)].copy(); uji = Dp[Dp.tahun == u].copy()
        for x, ks in ((latih, True), (uji, False)):
            gg, tt, mm = x._g.to_numpy(int), x._t.to_numpy(int), x._m.to_numpy(int)
            x["anomali_nasional"] = anomali(tl, tt, mm, keluar_sendiri=ks)
            x["anomali_dc"] = anomali_dc(tl, gg, tt, mm, keluar_sendiri=ks)
        xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                           reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        xg.fit(latih[FIT].fillna(0.0).to_numpy(), latih[SAS].to_numpy())
        b = uji[["gid", "bulan"]].copy()
        b["sasaran"], b["skema"], b["tag"] = SAS, "S1", str(u)
        b["xgb_nasdc"] = xg.predict_proba(uji[FIT].fillna(0.0).to_numpy())[:, 1]
        baris.append(b)
        print(f"O1 {u}: dilatih {len(latih):,} baris, {len(FIT)} fitur", flush=True)
    pd.concat(baris, ignore_index=True).to_csv(OUTGB, index=False)
    del Dp

# ---------------------------------------------------------------- prediksi tersimpan (seperti fase L)
P = B.muat_P(fwi=False)
P = P[(P.sasaran == SAS) & (P.skema == "S1") & P.tag.isin(B.BIASA)]
for f, kol in ((f"patokan_revisi_peluang_{SAS}.csv", ["rujukan_peluang", "ef1_tingkat"]),
               ("patokan_revisi_sederhana.csv", ["log_ef1_jeda"]), ("patokan_revisi_setel.csv", ["xgb_setel"]),
               (f"patokan_revisi_tingkat_gb_{SAS}.csv", ["xgb_nas"]),
               (os.path.basename(OUTGB), ["xgb_nasdc"])):
    X = pd.read_csv(os.path.join(B.D, f), dtype={"tag": str})
    X = X[(X.sasaran == SAS) & (X.skema == "S1") & X.tag.isin(B.BIASA)][KUNCI + kol]
    P = P.merge(X, on=KUNCI, how="left", validate="one_to_one")
P = P.merge(BARU, on=KUNCI, how="left", validate="one_to_one")
g = B.ambil_sasaran(P, SAS).copy().reset_index(drop=True)
del P, BARU
MODEL = ["log_ef1_jeda", "rf", "xgb", "xgb_setel", "mlp"]
KOL = [REF, "rujukan_peluang", "ef1_tingkat"] + MODEL + ["xgb_nas", "xgb_nasdc", "efl_ulang", "efd", "efld"]
assert not g[KOL].isna().any().any(), g[KOL].isna().sum()
beda = np.abs(g.efl_ulang - g.ef1_tingkat).max()
print(f"\n{SAS}: rujukan pasangan {B.LABEL[REF]}; kabupaten {g.gid.nunique()}; baris {len(g):,}; "
      f"EF-L ulang lawan tersimpan, selisih maks {beda:.2e}", flush=True)
assert beda < 1e-4, beda

tags = list(B.BIASA)
kab = g.gid.unique()
pos = {k: i for i, k in enumerate(kab)}
ix = {c: i for i, c in enumerate(KOL)}
S = np.zeros((len(tags), len(kab), len(KOL)))
for a, t in enumerate(tags):
    x = g[g.tag == t]
    j = x.gid.map(pos).to_numpy(); y = x.y.to_numpy()
    for c in KOL:
        S[a, :, ix[c]] = np.bincount(j, weights=(x[c].to_numpy() - y) ** 2, minlength=len(kab))
th = S.sum(1)


def dua_tahap_cepat(n, rng):
    nt, nk = S.shape[0], S.shape[1]
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        pilih = rng.choice(nk, nk)
        tt = rng.integers(0, nt, nt)
        out[r] = S[:, pilih, :].sum(1)[tt]
    return out


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def sel(R, a, r):
    pt = 1 - th[:, ix[a]] / th[:, ix[r]]
    rr = np.percentile((1 - R[:, :, ix[a]] / R[:, :, ix[r]]).mean(1), [2.5, 97.5])
    sr = 1 - th[:, ix[a]].sum() / th[:, ix[r]].sum()
    srq = np.percentile(1 - R[:, :, ix[a]].sum(1) / R[:, :, ix[r]].sum(1), [2.5, 97.5])
    return pt, rr, sr, srq


R0 = dua_tahap_cepat(1000, np.random.default_rng(B.SEED))
SEL_O1 = [("xgb", "efld"), ("xgb_nasdc", "efld"), ("xgb_nasdc", "ef1_tingkat"), ("xgb_nasdc", REF),
          ("xgb_nasdc", "rujukan_peluang"), ("xgb_nasdc", "xgb_nas"), ("xgb_nasdc", "xgb"), ("xgb_nas", "efld")]
print("\nO1 (a) rata-rata rasio [2 tahap] t95; (b) satu rasio [2 tahap]; (c) median, tahun > 0 (n = 6)")
for a, r in SEL_O1:
    pt, rr, sr, srq = sel(R0, a, r)
    tl, tu = t95(pt)
    lolos = rr[0] > 0 and tl > 0
    print(f"  {a:<13} atas {r:<16}(a) {pt.mean():+.4f} [{rr[0]:+.4f}, {rr[1]:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
          f"{'  LOLOS' if lolos else ''}  (b) {sr:+.4f} [{srq[0]:+.4f}, {srq[1]:+.4f}]  (c) {np.median(pt):+.4f} "
          f"{int((pt > 0).sum())}/6  per tahun " + " ".join(f"{v:+.3f}" for v in pt), flush=True)

print("\nO1 galat Monte Carlo: batas dua tahap per jalan (rata-rata rasio | satu rasio)")
JALAN = [(f"benih {b}", 1000, b) for b in (B.SEED, 1, 2, 3, 4)] + [("10.000 ulangan", 10000, B.SEED)]
SEL_MC = [("xgb_nasdc", "efld"), ("xgb_nasdc", "ef1_tingkat"), ("xgb_nasdc", REF), ("xgb_nasdc", "rujukan_peluang")]
HAS = {}
for nama, n, b in JALAN:
    Rj = R0 if (n == 1000 and b == B.SEED) else dua_tahap_cepat(n, np.random.default_rng(b))
    for a, r in SEL_MC:
        _, rr, _, srq = sel(Rj, a, r)
        HAS.setdefault((a, r), []).append((rr, srq))
    print(f"  {nama} selesai", flush=True)
for (a, r), v in HAS.items():
    v1 = [x[0] for x in v]; v2 = [x[1] for x in v]
    print(f"  {a:<13} atas {r:<16} rata-rata rasio bawah {min(x[0] for x in v1):+.4f}..{max(x[0] for x in v1):+.4f} "
          f"semua > 0: {all(x[0] > 0 for x in v1)} | satu rasio bawah {min(x[0] for x in v2):+.4f}.."
          f"{max(x[0] for x in v2):+.4f} semua > 0: {all(x[0] > 0 for x in v2)}", flush=True)
print("selesai.")
