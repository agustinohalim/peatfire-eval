"""
Revisi fase M (Hasil_Patokan_18), panel VIIRS, tahun biasa 2020-2025 (skema S1), satu sasaran per jalan.
  M1  rujukan kekeringan bentuk tetap, dipasang per lipatan di baris tahun latih seperti EF-L
      (patokan_revisi_peluang.py): EF-D = logistik logit(ef1_halus) + anomali log1p(DC bulan lalu) kabupaten;
      EF-LD = logistik logit(ef1_halus) + anomali api nasional bulan lalu + anomali DC kabupaten. Anomali DC =
      log1p(DC bulan lalu) - rata-rata kabupaten itu pada bulan kalender sama di tahun latih (baris latih tanpa
      tahunnya sendiri). DC dari gfwed_komponen_MERRA2.CORRECTED.csv, digeser satu bulan per kabupaten, hilang
      diisi median kabupaten (sama dengan fase D). EF-L dihitung ulang dan harus sama dengan ef1_tingkat tersimpan.
      Sel: EF-D, EF-LD atas EF-L / kalender / EF-P; GB, RF atas EF-LD; log_ef1_fdrs, ef1_cuaca atas EF-L;
      agregasi (a) rata-rata rasio tahunan dua tahap + t95, (b) satu rasio dua tahap, (c) median, tahun > 0.
  M2c sel tabel L1 Hasil_Patokan_17 (dan sel M1): dua tahap lima benih {B.SEED, 1, 2, 3, 4} x 1.000 dan 10.000
      ulangan benih B.SEED; ditandai sel yang vonis dua tahapnya berubah.

Pakai:
    python -u patokan_tingkat_m.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_m_kab.log
    python -u patokan_tingkat_m.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_m_nas.log
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B

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

# ---------------------------------------------------------------- prediksi tersimpan (seperti fase L)
P = B.muat_P(fwi=False)
P = P[(P.sasaran == SAS) & (P.skema == "S1") & P.tag.isin(B.BIASA)]
for f, kol in ((f"patokan_revisi_peluang_{SAS}.csv", ["rujukan_peluang", "ef1_tingkat"]),
               ("patokan_revisi_sederhana.csv", ["log_ef1_jeda"]), ("patokan_revisi_setel.csv", ["xgb_setel"]),
               (f"patokan_revisi_tingkat_gb_{SAS}.csv", ["xgb_nas"]),
               ("patokan_revisi_fdrs.csv", ["log_ef1_fdrs", "ef1_cuaca"])):
    X = pd.read_csv(os.path.join(B.D, f), dtype={"tag": str})
    X = X[(X.sasaran == SAS) & (X.skema == "S1") & X.tag.isin(B.BIASA)][KUNCI + kol]
    P = P.merge(X, on=KUNCI, how="left", validate="one_to_one")
P = P.merge(BARU, on=KUNCI, how="left", validate="one_to_one")
g = B.ambil_sasaran(P, SAS).copy().reset_index(drop=True)
del P, BARU
MODEL = ["log_ef1_jeda", "rf", "xgb", "xgb_setel", "mlp"]
KOL = [REF, "rujukan_peluang", "ef1_tingkat"] + MODEL + ["xgb_nas", "log_ef1_fdrs", "ef1_cuaca", "efl_ulang", "efd", "efld"]
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
SEL_M1 = [(c, r) for c in ("efd", "efld") for r in ("ef1_tingkat", REF, "rujukan_peluang")] + \
         [("xgb", "efld"), ("rf", "efld"), ("log_ef1_fdrs", "ef1_tingkat"), ("ef1_cuaca", "ef1_tingkat")]
SEL_L1 = [("xgb", REF), ("rf", REF), ("ef1_tingkat", REF), ("xgb", "rujukan_peluang")] + \
         [(c, "ef1_tingkat") for c in ("xgb", "xgb_setel", "rf", "log_ef1_jeda", "mlp", "xgb_nas")]
print("\nM1 (a) rata-rata rasio [2 tahap] t95; (b) satu rasio [2 tahap]; (c) median, tahun > 0 (n = 6)")
for a, r in SEL_M1:
    pt, rr, sr, srq = sel(R0, a, r)
    tl, tu = t95(pt)
    lolos = rr[0] > 0 and tl > 0
    print(f"  {a:<13} atas {r:<16}(a) {pt.mean():+.4f} [{rr[0]:+.4f}, {rr[1]:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
          f"{'  LOLOS' if lolos else ''}  (b) {sr:+.4f} [{srq[0]:+.4f}, {srq[1]:+.4f}]  (c) {np.median(pt):+.4f} "
          f"{int((pt > 0).sum())}/6  per tahun " + " ".join(f"{v:+.3f}" for v in pt), flush=True)

print("\nM2c galat Monte Carlo: batas dua tahap per jalan (rata-rata rasio | satu rasio)")
JALAN = [(f"benih {b}", 1000, b) for b in (B.SEED, 1, 2, 3, 4)] + [("10.000 ulangan", 10000, B.SEED)]
HAS = {}
for nama, n, b in JALAN:
    R = R0 if (n == 1000 and b == B.SEED) else dua_tahap_cepat(n, np.random.default_rng(b))
    for a, r in SEL_L1 + SEL_M1:
        _, rr, _, srq = sel(R, a, r)
        HAS.setdefault((a, r), []).append((rr, srq))
    print(f"  {nama} selesai", flush=True)
for (a, r), v in HAS.items():
    v1 = [x[0] for x in v]; v2 = [x[1] for x in v]
    vo1 = {lo > 0 or hi < 0 for lo, hi in v1}; vo2 = {lo > 0 or hi < 0 for lo, hi in v2}
    print(f"  {a:<13} atas {r:<16} rata-rata rasio bawah {min(x[0] for x in v1):+.4f}..{max(x[0] for x in v1):+.4f} "
          f"atas {min(x[1] for x in v1):+.4f}..{max(x[1] for x in v1):+.4f}{'  VONIS BERUBAH' if len(vo1) > 1 else ''} | "
          f"satu rasio bawah {min(x[0] for x in v2):+.4f}..{max(x[0] for x in v2):+.4f} "
          f"atas {min(x[1] for x in v2):+.4f}..{max(x[1] for x in v2):+.4f}{'  VONIS BERUBAH' if len(vo2) > 1 else ''}",
          flush=True)
print("selesai.")
