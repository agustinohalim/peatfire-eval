"""
Revisi fase R2 (Hasil_Patokan_24): Tabel 4 (VIIRS S1 2020-2025, ambang seluruh panel) dengan tahap pertama dua tahap
mengambil ulang provinsi alih-alih kabupaten. Bagian data disalin dari patokan_tingkat_o.py tanpa perubahan isi.

Pakai:
    python -u patokan_tingkat_r.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_r_kab.log
    python -u patokan_tingkat_r.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_r_nas.log
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


def dua_tahap(n, rng, grup=None):
    nt, nk = S.shape[0], S.shape[1]
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        if grup is None:
            pilih = rng.choice(nk, nk)
        else:
            pr = rng.integers(0, len(grup), len(grup))
            pilih = np.concatenate([grup[i] for i in pr])
        tt = rng.integers(0, nt, nt)
        out[r] = S[:, pilih, :].sum(1)[tt]
    return out


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


pv = pd.read_csv(f"{B.D}/panel_nasional_bersih.csv", usecols=["gid", "provinsi"]).drop_duplicates()
peta = dict(zip(pv.gid, pv.provinsi))
nama = sorted({peta[k] for k in kab})
grup = [np.array([pos[k] for k in kab if peta[k] == p_]) for p_ in nama]
print(f"\nR2 {SAS}: provinsi {len(grup)}, kabupaten {len(kab)}; 1.000 ulangan benih {B.SEED}", flush=True)
RK = dua_tahap(1000, np.random.default_rng(B.SEED))
RP = dua_tahap(1000, np.random.default_rng(B.SEED), grup)
RUJ = [(REF, "kalender"), ("rujukan_peluang", "EF-P"), ("ef1_tingkat", "EF-L"), ("efld", "EF-LD")]
R10 = None
BARIS = ["rujukan_peluang", "ef1_tingkat", "efd", "efld"] + MODEL + ["xgb_nas", "xgb_nasdc"]
for a in BARIS:
    for r, nr in RUJ:
        if a == r or (a in ("rujukan_peluang", "ef1_tingkat") and r != REF):
            continue
        pt = 1 - th[:, ix[a]] / th[:, ix[r]]
        dk = (1 - RK[:, :, ix[a]] / RK[:, :, ix[r]]).mean(1)
        dp = (1 - RP[:, :, ix[a]] / RP[:, :, ix[r]]).mean(1)
        tl, tu = t95(pt)
        lo = np.percentile(dp, 2.5)
        teks = ""
        if abs(lo) <= 0.01:
            if R10 is None:
                R10 = dua_tahap(10000, np.random.default_rng(B.SEED), grup)
            d10 = (1 - R10[:, :, ix[a]] / R10[:, :, ix[r]]).mean(1)
            teks = f"  provinsi 10.000 [{np.percentile(d10, 2.5):+.4f}, {np.percentile(d10, 97.5):+.4f}]"
        print(f"  {a:<14} atas {nr:<9} {pt.mean():+.4f}  kabupaten [{np.percentile(dk, 2.5):+.4f}, {np.percentile(dk, 97.5):+.4f}]  "
              f"provinsi [{lo:+.4f}, {np.percentile(dp, 97.5):+.4f}]  tahun t [{tl:+.4f}, {tu:+.4f}]{teks}", flush=True)
print("selesai.")
