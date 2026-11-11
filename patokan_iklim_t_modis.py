"""
Revisi fase T (Hasil_Patokan_26), jalur MODIS: EF-E dan EF-LE pada prakiraan maju 2008-2019 (lipatan R fase N: latih =
tahun MODIS sebelum tahun uji, ambang dan kabupaten dari tahun latih, lantai bawaan), satu sasaran per jalan.

  EF-E : logit P = b0 + b1 logit(ef1_halus) + b2 ONI_m-1 + b3 DMI_m-1
  EF-LE: logit P = b0 + b1 logit(ef1_halus) + b2 anomali nasional + b3 ONI_m-1 + b4 DMI_m-1
ONI dan DMI dibaca dengan muat_oni / muat_dmi dari gabung_indeks_nasional.py (berkas sama dengan panel VIIRS); ONI_m-1
adalah musim tiga bulan yang berakhir di bulan m-1.

Pengaturan panel, label, larik, rujukan dan anomali disalin dari patokan_modis_n.py (baris ditandai). Kalender, EF-L,
EF-LD dan GB-FH (rata-rata enam benih) diambil dari DL_FIRE_MODIS/sse_modis_n_<nas|kab>.npz; SSE EF-L yang dihitung
ulang di sini harus sama dengan kolom efl berkas itu (selisih relatif < 1e-6). Selang: dua tahap cepat fase N
(1.000 ulangan, benih B.SEED) dan selang tahun t (df 11).

Pakai:
    python -u patokan_iklim_t_modis.py y_kabupaten > DL_FIRE_MODIS/patokan_iklim_t_modis_kab.log
    python -u patokan_iklim_t_modis.py y_gabungan  > DL_FIRE_MODIS/patokan_iklim_t_modis_nas.log
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B
from gabung_indeks_nasional import muat_dmi, muat_oni

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
KAPPA = 2.0
N = 1000
DM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_MODIS")
IDX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_SV-C2_792597")
NM = "nas" if SAS == "y_gabungan" else "kab"

# ---- disalin dari patokan_modis_n.py:41-56 dan 72-74 tanpa perubahan isi (LANTAI = None)
V = pd.read_csv(os.path.join(B.D, "panel_nasional_bersih.csv"), usecols=["gid", "tahun", "titik_panas"])
M = pd.read_csv(os.path.join(DM, "panel_modis.csv"), usecols=["gid", "bulan", "tahun", "bulan_ke", "titik_panas"])
M = M[M.gid.isin(V.gid.unique()) & M.tahun.between(2003, 2025)].sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
vm = V[V.tahun.between(2012, 2025)].titik_panas.sum(); mm = M[M.tahun.between(2012, 2025)].titik_panas.sum()
r = mm / vm; F0 = max(1, int(round(10 * r)))
F = F0
GID = np.sort(M.gid.unique()); THN = np.arange(2003, 2026)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
G_, T_, M_ = M.gid.map(gi).to_numpy(), M.tahun.map(ti).to_numpy(), (M.bulan_ke - 1).to_numpy()
TP = np.full((len(GID), len(THN), 12), np.nan)
TP[G_, T_, M_] = M.titik_panas.to_numpy(float)
NB = len(THN) * 12
TPf = TP.reshape(len(GID), NB)
del V, M
nas = np.nansum(TPf, 0)
LNf = np.full(NB, np.nan); LNf[1:] = np.log1p(nas[:-1])
LN = LNf.reshape(len(THN), 12)
# ---- akhir salinan

# ONI dan DMI bulan m-1 untuk sasaran (tahun, bulan m)
oni_map = muat_oni(os.path.join(IDX, "oni.ascii.txt"))
dmi_map = muat_dmi(os.path.join(IDX, "dmi.had.long.data"))
ONI = np.full((len(THN), 12), np.nan); DMI = np.full((len(THN), 12), np.nan)
for j, t in enumerate(THN):
    for m in range(12):
        p = pd.Period(f"{t}-{m + 1:02d}", freq="M") - 1
        ONI[j, m] = oni_map.get(p, np.nan); DMI[j, m] = dmi_map.get(p, np.nan)
print(f"MODIS F {F}; kabupaten {len(GID)}; sasaran {SAS}; ONI/DMI m-1 2003-2019 hilang "
      f"{int(np.isnan(ONI[:17]).sum())}/{int(np.isnan(DMI[:17]).sum())}", flush=True)


# ---- disalin dari patokan_modis_n.py:91-197 (label, larik, jendela, mc, rujukan, anomali, logit, indeks) tanpa
# perubahan isi; anomali_dc dan X_gb tidak dipakai
def label(tahun_ambang):
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
# ---- akhir salinan


def iklim(t, m):
    o, d = ONI[t, m], DMI[t, m]
    assert not np.isnan(o).any() and not np.isnan(d).any(), "ONI/DMI hilang"
    return o, d


def lipatan(u):
    """Baris latih dan uji sama dengan patokan_modis_n.py:lipatan (lipatan R: tl = tahun < u, ambang tahun latih)."""
    tl = [x for x in THN if x < u]
    L, ada = label(tl)
    Y = larik(L, ada)
    g, t, m = indeks(Y, tl)
    h = logit(rujukan(Y, tl, g, t, m, keluar_sendiri=True)["ef1_halus"])
    a = anomali(tl, t, m, keluar_sendiri=True)
    o, d = iklim(t, m)
    yl = Y[g, t, m]
    fit = lambda *X: LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack(X), yl)
    mL, mE, mLE = fit(h, a), fit(h, o, d), fit(h, a, o, d)
    g2, t2, m2 = indeks(Y, [u])
    h, a = logit(rujukan(Y, tl, g2, t2, m2)["ef1_halus"]), anomali(tl, t2, m2)
    o, d = iklim(t2, m2)
    print(f"  {u}: latih {len(tl)} thn {len(yl):,} baris | EF-L b {mL.coef_[0, 0]:+.3f} nas {mL.coef_[0, 1]:+.3f} | "
          f"EF-E b {mE.coef_[0, 0]:+.3f} ONI {mE.coef_[0, 1]:+.3f} DMI {mE.coef_[0, 2]:+.3f} | EF-LE nas "
          f"{mLE.coef_[0, 1]:+.3f} ONI {mLE.coef_[0, 2]:+.3f} DMI {mLE.coef_[0, 3]:+.3f}", flush=True)
    return pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2],
                         "efl_t": mL.predict_proba(np.column_stack([h, a]))[:, 1],
                         "efe": mE.predict_proba(np.column_stack([h, o, d]))[:, 1],
                         "efle": mLE.predict_proba(np.column_stack([h, a, o, d]))[:, 1]})


uji = list(range(2008, 2020))
tags = [str(u) for u in uji]
P = pd.concat([lipatan(u) for u in uji], ignore_index=True)
kab = P.gid.unique(); pos = {k: i for i, k in enumerate(kab)}
BARU = ["efl_t", "efe", "efle"]
S2 = np.zeros((len(tags), len(kab), len(BARU)))
for a, t in enumerate(tags):
    x = P[P.tag == t]
    j = x.gid.map(pos).to_numpy(); y = x.y.to_numpy()
    for c, k in enumerate(BARU):
        S2[a, :, c] = np.bincount(j, weights=(x[k].to_numpy() - y) ** 2, minlength=len(kab))

z = np.load(os.path.join(DM, f"sse_modis_n_{NM}.npz"), allow_pickle=True)
assert list(z["tags"]) == tags and np.array_equal(z["kab"].astype(str), kab.astype(str)), "urutan kabupaten/tahun beda"
kel = list(z["keluaran"])
LAMA = ["kalender", "efp", "efl", "efld", "gb"]
S1 = z["S"][:, :, [kel.index(k) for k in LAMA]]
rel = np.abs(S2[:, :, 0].sum(1) - S1[:, :, 2].sum(1)) / S1[:, :, 2].sum(1)
print(f"cek SSE EF-L dihitung ulang vs fase N: selisih relatif maks per tahun {rel.max():.2e}", flush=True)
assert rel.max() < 1e-6
S = np.concatenate([S1, S2[:, :, 1:]], axis=2)
ok = {k: i for i, k in enumerate(LAMA + ["efe", "efle"])}
th = S.sum(1)


# ---- disalin dari patokan_modis_n.py:236-266 (t95, dua_tahap_cepat) tanpa perubahan isi
def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def dua_tahap_cepat(S, n, rng):
    nt, nk, _ = S.shape
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        pilih = rng.choice(nk, nk)
        tt = rng.integers(0, nt, nt)
        cs = S[:, pilih, :].sum(1)
        out[r] = cs[tt]
    return out
# ---- akhir salinan


R = dua_tahap_cepat(S, N, np.random.default_rng(B.SEED))
NAMA = {"kalender": "kalender", "efp": "EF-P", "efl": "EF-L", "efld": "EF-LD", "gb": "GB-FH", "efe": "EF-E", "efle": "EF-LE"}
SEL = [("efe", "kalender"), ("efe", "efl"), ("efl", "efe"), ("efle", "efl"), ("efle", "efe"), ("gb", "efe"),
       ("efld", "efe"), ("efe", "efld"), ("gb", "efl"), ("efl", "kalender")]
print(f"\nN R {SAS}, prakiraan maju 2008-2019, kabupaten {len(kab)}")
print(f"{'prakiraan':<10}{'atas':<10}{'BSS':>8}  {'[dua tahap]':<20}{'[tahun t]':<20}{'vonis':<8}"
      f"{'satu rasio':>10} {'median':>7} {'thn>0':>6}  per tahun")
for a, r in SEL:
    pt = 1 - th[:, ok[a]] / th[:, ok[r]]
    lo, hi = np.percentile((1 - R[:, :, ok[a]] / R[:, :, ok[r]]).mean(1), [2.5, 97.5])
    tl, tu = t95(pt)
    sr = 1 - th[:, ok[a]].sum() / th[:, ok[r]].sum()
    v = "UNGGUL" if lo > 0 and tl > 0 else ("KALAH" if hi < 0 and tu < 0 else "")
    print(f"{NAMA[a]:<10}{NAMA[r]:<10}{pt.mean():>+8.3f}  [{lo:+.3f}, {hi:+.3f}]    [{tl:+.3f}, {tu:+.3f}]    {v:<8}"
          f"{sr:>+10.3f} {np.median(pt):>+7.3f} {int((pt > 0).sum()):>3}/12  " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
print("selesai.")
