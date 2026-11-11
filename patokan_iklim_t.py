"""
Revisi fase T (Hasil_Patokan_26), jalur VIIRS: rujukan keadaan iklim EF-E dan EF-LE, S1 2020-2025, ambang tahun latih,
satu sasaran per jalan.

  EF-E : logit P = b0 + b1 logit(ef1_halus) + b2 ONI_m-1 + b3 DMI_m-1
  EF-LE: logit P = b0 + b1 logit(ef1_halus) + b2 anomali nasional + b3 ONI_m-1 + b4 DMI_m-1
ONI_m-1 = kolom oni_lag1, DMI_m-1 = kolom dmi_lag1 dari panel fitur (konvensi gabung_indeks_nasional.py), mentah.

Label, larik Y dan fungsi EF disalin dari patokan_ambang_s.py (baris ditandai). EF-L dihitung ulang dengan rumus yang
sama dan harus sama dengan kolom ef1_tingkat di DL_FIRE_NASIONAL/patokan_ambang_s_<sasaran>.csv (< 1e-6); model belajar
dan EF-LD diambil dari berkas itu tanpa dilatih ulang. Skor dan selang sama dengan patokan_ambang_s_tabel.py (S2):
BSS dalam-tahun rata-rata rasio, dua tahap kabupaten-tahun 1.000 ulangan benih B.SEED, selang tahun t (df 5).

Pakai:
    python -u patokan_iklim_t.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_iklim_t_kab.log
    python -u patokan_iklim_t.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_iklim_t_nas.log
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B
from kelompok_fitur import PANEL

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
assert SAS in B.PASANGAN, SAS
REF = B.PASANGAN[SAS]
UJI = list(range(2020, 2026))
KAPPA = 2.0
BATAS_BAWAH = 10
N = 1000

df = pd.read_csv(os.path.join(B.D, PANEL), usecols=["gid", "tahun", "bulan", "bulan_ke", "titik_panas", "tp_lag1",
                                                   "oni_lag1", "dmi_lag1"])
# ---- disalin dari patokan_ambang_s.py:74-88 (label_ulang) tanpa perubahan isi
panel = pd.read_csv(f"{B.D}/panel_nasional_bersih.csv", usecols=["gid", "tahun", "titik_panas"])
semua = sorted(panel.tahun.unique())


def label_ulang(d, tahun_ambang):
    ref = panel[panel.tahun.isin(tahun_ambang)]
    q_kab = np.maximum(ref.groupby("gid").titik_panas.quantile(0.90), BATAS_BAWAH)
    gab = ref.titik_panas.quantile(0.90)
    d = d.copy()
    d["y_kabupaten"] = (d.titik_panas > d.gid.map(q_kab)).astype(int)
    d["y_gabungan"] = (d.titik_panas > gab).astype(int)
    return d
# ---- akhir salinan


# ---- disalin dari patokan_ambang_s.py:93-99 dan 112 (indeks larik, LN) tanpa perubahan isi
GID = np.sort(df.gid.unique()); THN = np.arange(df.tahun.min(), df.tahun.max() + 1)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
df["_g"] = df.gid.map(gi); df["_t"] = df.tahun.map(ti); df["_m"] = df.bulan_ke - 1
NAS = df.groupby(["_t", "_m"]).tp_lag1.sum()
LN = np.full((len(THN), 12), np.nan)
LN[NAS.index.get_level_values(0), NAS.index.get_level_values(1)] = np.log1p(NAS.to_numpy())
G_, T_, M_ = df.gid.map(gi).to_numpy(), df.tahun.map(ti).to_numpy(), (df.bulan_ke - 1).to_numpy()
# ---- akhir salinan

# ONI dan DMI bulan lalu (nasional): satu nilai per (tahun, bulan)
IK = df.groupby(["_t", "_m"])[["oni_lag1", "dmi_lag1"]].agg(["min", "max"])
for k in ("oni_lag1", "dmi_lag1"):
    assert (IK[(k, "min")] == IK[(k, "max")]).all(), f"{k} tidak tunggal per bulan"
ONI = np.full((len(THN), 12), np.nan); DMI = np.full((len(THN), 12), np.nan)
ONI[IK.index.get_level_values(0), IK.index.get_level_values(1)] = IK[("oni_lag1", "min")].to_numpy()
DMI[IK.index.get_level_values(0), IK.index.get_level_values(1)] = IK[("dmi_lag1", "min")].to_numpy()


# ---- disalin dari patokan_ambang_s.py:118-171 (jendela, rujukan, anomali, logit, indeks) tanpa perubahan isi
def jendela(a):
    return a + np.roll(a, 1, axis=-1) + np.roll(a, -1, axis=-1)


def rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=False):
    """rujukan untuk baris (g, t, m) dari tahun latih; keluar_sendiri: buang tahun t baris itu (baris latih)."""
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
        yo = Y[g, t, m]; ada = ~np.isnan(yo); yo0 = np.where(ada, yo, 0.0)
        s1 = s1 - yo0; n1 = n1 - ada
        w = np.stack([Y[g, t, (m - 1) % 12], yo, Y[g, t, (m + 1) % 12]])
        s3 = s3 - np.nansum(w, 0); n3 = n3 - np.sum(~np.isnan(w), 0)
        dlm_pasca = THN[t] >= 2016
        sp = sp - np.where(dlm_pasca, yo0, 0); np_ = np_ - np.where(dlm_pasca, ada, 0)
        sm = sm - yo0; nm = nm - ada; smb = smb - yo0; nmb = nmb - ada
    bagi = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), 0.0)
    out = {"konstan": bagi(sm, nm), "konstan_bulan": bagi(smb, nmb), "ef1": bagi(s1, n1), "ef3": bagi(s3, n3)}
    out["ef1_pasca"] = np.where(np_ > 0, bagi(sp, np_), out["konstan_bulan"])
    out["ef1_halus"] = (s1 + KAPPA * out["ef3"]) / (n1 + KAPPA)
    return out


def anomali(tahun_latih, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(LN[msk], 0); n = np.sum(~np.isnan(LN[msk]), 0).astype(float)
    s, n = s[m], n[m]
    if keluar_sendiri:
        s = s - LN[t, m]; n = n - 1
    return LN[t, m] - s / np.maximum(n, 1)


def logit(p, e=1e-4):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def indeks(Y, tahun):
    g, t, m = np.where(~np.isnan(Y[:, np.isin(THN, tahun), :]))
    t = np.flatnonzero(np.isin(THN, tahun))[t]
    return g, t, m
# ---- akhir salinan


def iklim(t, m):
    o, d = ONI[t, m], DMI[t, m]
    assert not np.isnan(o).any() and not np.isnan(d).any(), \
        f"ONI/DMI hilang: {sorted(set(zip(THN[t][np.isnan(o) | np.isnan(d)], m[np.isnan(o) | np.isnan(d)] + 1)))}"
    return o, d


def lipatan_t(Y, u):
    """EF-L (cek), EF-E, EF-LE satu lipatan S1; baris latih dan uji sama dengan efl_lipatan (patokan_ambang_s.py:226-240)."""
    tl = [x for x in THN if x < u]
    g, t, m = indeks(Y, tl)
    h = logit(rujukan(Y, tl, g, t, m, keluar_sendiri=True)["ef1_halus"])
    a = anomali(tl, t, m, keluar_sendiri=True)
    o, d = iklim(t, m)
    yl = Y[g, t, m]
    fit = lambda *X: LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack(X), yl)
    mL, mE, mLE = fit(h, a), fit(h, o, d), fit(h, a, o, d)
    g, t, m = indeks(Y, [u])
    h, a = logit(rujukan(Y, tl, g, t, m)["ef1_halus"]), anomali(tl, t, m)
    o, d = iklim(t, m)
    print(f"  {u}: latih {len(yl):,} baris | EF-L b {mL.coef_[0, 0]:+.3f} nas {mL.coef_[0, 1]:+.3f} | "
          f"EF-E b {mE.coef_[0, 0]:+.3f} ONI {mE.coef_[0, 1]:+.3f} DMI {mE.coef_[0, 2]:+.3f} | EF-LE b {mLE.coef_[0, 0]:+.3f} "
          f"nas {mLE.coef_[0, 1]:+.3f} ONI {mLE.coef_[0, 2]:+.3f} DMI {mLE.coef_[0, 3]:+.3f}", flush=True)
    return pd.DataFrame({"tag": str(u), "gid": GID[g], "bulan": [f"{THN[p]}-{q + 1:02d}" for p, q in zip(t, m)],
                         "efl_t": mL.predict_proba(np.column_stack([h, a]))[:, 1],
                         "efe": mE.predict_proba(np.column_stack([h, o, d]))[:, 1],
                         "efle": mLE.predict_proba(np.column_stack([h, a, o, d]))[:, 1]})


print(f"{SAS}: fase T VIIRS, kalender {B.LABEL[REF]}; panel {len(df):,} baris", flush=True)
BARIS = []
for u in UJI:
    d = label_ulang(df, [t for t in semua if t < u])
    pos = d[d.tahun < u].groupby("gid")[SAS].sum()
    # ---- disalin dari patokan_ambang_s.py:550-553 tanpa perubahan isi
    Y = np.full((len(GID), len(THN), 12), np.nan)
    Y[G_, T_, M_] = d[SAS].to_numpy(float)
    if SAS == "y_kabupaten":
        Y[~np.isin(GID, pos[pos > 0].index)] = np.nan
    # ---- akhir salinan
    BARIS.append(lipatan_t(Y, u))
T = pd.concat(BARIS, ignore_index=True)

KT = ["tag", "gid", "bulan"]
MODEL = ["rf", "xgb", "xgb_setel", "mlp", "xgb_nas", "xgb_nasdc"]
U = pd.read_csv(os.path.join(B.D, f"patokan_ambang_s_{SAS}.csv"), dtype={"tag": str},
                usecols=KT + ["y", REF, "rujukan_peluang", "ef1_tingkat", "efld"] + MODEL)
n0 = len(U)
U = U.merge(T, on=KT, how="left", validate="one_to_one")
assert len(U) == n0 and not U[["efl_t", "efe", "efle"]].isna().any().any(), "baris uji tidak cocok dengan fase S"
dmax = np.abs(U.efl_t - U.ef1_tingkat).max()
print(f"cek EF-L dihitung ulang vs fase S: selisih maks {dmax:.2e}", flush=True)
assert dmax < 1e-6
print(f"baris uji {len(U):,}, kabupaten {U.gid.nunique()}, positif {int(U.y.sum()):,}", flush=True)


# ---- disalin dari patokan_ambang_s_tabel.py:58-66 dan 147-176 tanpa perubahan isi (KOL diganti)
def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def tanpa_nol(lo, hi):
    return lo > 0 or hi < 0


KOL = [REF, "rujukan_peluang", "ef1_tingkat", "efld", "efe", "efle"] + MODEL
g = U.reset_index(drop=True)
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


def sel(R, a, r):
    pt = 1 - th[:, ix[a]] / th[:, ix[r]]
    rr = np.percentile((1 - R[:, :, ix[a]] / R[:, :, ix[r]]).mean(1), [2.5, 97.5])
    sr = 1 - th[:, ix[a]].sum() / th[:, ix[r]].sum()
    srq = np.percentile(1 - R[:, :, ix[a]].sum(1) / R[:, :, ix[r]].sum(1), [2.5, 97.5])
    return pt, rr, sr, srq
# ---- akhir salinan


RB = dua_tahap_cepat(N, np.random.default_rng(B.SEED))
NAMA = {REF: "kalender", "rujukan_peluang": "EF-P", "ef1_tingkat": "EF-L", "efld": "EF-LD", "efe": "EF-E", "efle": "EF-LE",
        "rf": "RF", "xgb": "GB", "xgb_setel": "GB tersetel", "mlp": "MLP", "xgb_nas": "GB + nasional",
        "xgb_nasdc": "GB + nasional + DC"}
NB = np.array([(g.tag == t).sum() for t in tags])
print("Brier rata-rata tahun: " + "  ".join(f"{NAMA[c]} {(th[:, ix[c]] / NB).mean():.5f}"
                                         for c in (REF, "ef1_tingkat", "efe", "efle", "efld")))
SEL = ([("efe", REF), ("efe", "ef1_tingkat"), ("ef1_tingkat", "efe"), ("efle", "ef1_tingkat"), ("efle", "efe")]
       + [(m, "efe") for m in MODEL] + [("efld", "efe"), ("efe", "efld")])
print(f"\n{'prakiraan':<22}{'atas':<10}{'BSS':>8}  {'[dua tahap]':<20}{'[tahun t]':<20}{'vonis':<8}"
      f"{'satu rasio':>10} {'median':>7} {'thn>0':>5}  per tahun")
for a, r in SEL:
    pt, rr, sr, _ = sel(RB, a, r)
    tl, tu = t95(pt)
    v = "UNGGUL" if tanpa_nol(*rr) and tanpa_nol(tl, tu) and pt.mean() > 0 else (
        "KALAH" if tanpa_nol(*rr) and tanpa_nol(tl, tu) else "")
    print(f"{NAMA[a]:<22}{NAMA[r]:<10}{pt.mean():>+8.3f}  [{rr[0]:+.3f}, {rr[1]:+.3f}]    [{tl:+.3f}, {tu:+.3f}]    "
          f"{v:<8}{sr:>+10.3f} {np.median(pt):>+7.3f} {int((pt > 0).sum()):>3}/6  "
          + " ".join(f"{x:+.3f}" for x in pt), flush=True)
print("selesai.")
