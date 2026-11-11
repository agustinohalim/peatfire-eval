"""
Revisi fase U (Hasil_Patokan_27), jalur VIIRS: rujukan iklim musiman-wilayah EF-ES dan EF-LES, satu rasio gabungan,
klaster provinsi, Holm; S1 2020-2025, ambang tahun latih, satu sasaran per jalan.

  EF-ES : logit P = b0 + b1 logit(ef1_halus) + sum_k 1[k] (g_k ONI_m-1 + d_k DMI_m-1),
          k = (kemarau Jul-Nov | lain) x (Sumatra-Kalimantan | lain), tanpa intersep sel
  EF-LES: EF-ES + anomali nasional
EF-L, EF-E, EF-LE dihitung ulang dengan kode fase T (patokan_iklim_t.py); EF-L dicek terhadap fase S (< 1e-6). Model
belajar, EF-P, EF-LD dari DL_FIRE_NASIONAL/patokan_ambang_s_<sasaran>.csv tanpa dilatih ulang.

Pakai:
    python -u patokan_iklim_u.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_iklim_u_kab.log
    python -u patokan_iklim_u.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_iklim_u_nas.log
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


# ---------------------------------------------------------------- fase U (Hasil_Patokan_27)
SK = {"Aceh", "SumateraUtara", "SumateraBarat", "Riau", "KepulauanRiau", "Jambi", "Bengkulu", "SumateraSelatan",
      "BangkaBelitung", "Lampung", "KalimantanBarat", "KalimantanTengah", "KalimantanSelatan", "KalimantanTimur",
      "KalimantanUtara"}
_pv = pd.read_csv(f"{B.D}/panel_nasional_bersih.csv", usecols=["gid", "provinsi"]).drop_duplicates()
assert _pv.gid.is_unique, "gid dengan lebih dari satu provinsi"
PROV = dict(zip(_pv.gid, _pv.provinsi))
assert len(set(PROV.values())) == 34, sorted(set(PROV.values()))
assert SK <= set(PROV.values()), SK - set(PROV.values())


def musim_wilayah(gid, m, o, d):
    """Delapan kolom: ONI dan DMI dikali indikator sel (kemarau Jul-Nov | lain) x (Sumatra-Kalimantan | lain)."""
    assert all(x in PROV for x in np.unique(gid)), "gid tanpa provinsi"
    sk = np.array([PROV[x] in SK for x in gid])
    kem = (m >= 6) & (m <= 10)
    kol = []
    for a in (kem, ~kem):
        for b in (sk, ~sk):
            c = (a & b).astype(float)
            kol += [c * o, c * d]
    return np.column_stack(kol)


def grup_provinsi(kab):
    nama = sorted({PROV[k] for k in kab})
    return [np.array([i for i, k in enumerate(kab) if PROV[k] == p]) for p in nama]


def dua_tahap_provinsi(S, grup, n, rng):
    nt = S.shape[0]
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        pilih = np.concatenate([grup[i] for i in rng.integers(0, len(grup), len(grup))])
        tt = rng.integers(0, nt, nt)
        out[r] = S[:, pilih, :].sum(1)[tt]
    return out


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); k = len(p)
    adj = np.empty(k); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (k - i) * p[j])); adj[j] = run
    return adj


def nilai(th, RD, RP, a, r):
    """Satu sel: rata-rata rasio (titik, dua tahap kab, t, p, dua tahap prov) dan satu rasio (titik, kab, prov)."""
    pt = 1 - th[:, a] / th[:, r]
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    tl, tu = pt.mean() - q * se, pt.mean() + q * se
    p = 2 * stats.t.sf(abs(pt.mean() / se), n - 1) if se > 0 else 0.0
    rk = np.percentile((1 - RD[:, :, a] / RD[:, :, r]).mean(1), [2.5, 97.5])
    rp = np.percentile((1 - RP[:, :, a] / RP[:, :, r]).mean(1), [2.5, 97.5])
    sr = 1 - th[:, a].sum() / th[:, r].sum()
    sk = np.percentile(1 - RD[:, :, a].sum(1) / RD[:, :, r].sum(1), [2.5, 97.5])
    sp = np.percentile(1 - RP[:, :, a].sum(1) / RP[:, :, r].sum(1), [2.5, 97.5])
    return dict(pt=pt, m=pt.mean(), rk=rk, t=(tl, tu), p=p, rp=rp, sr=sr, sk=sk, sp=sp)


def vonis(lo, hi):
    return "+" if lo > 0 else ("-" if hi < 0 else "0")


def cetak(v, na, nr, padj=None):
    vm = "+" if (v["rk"][0] > 0 and v["t"][0] > 0) else ("-" if (v["rk"][1] < 0 and v["t"][1] < 0) else "0")
    vpm = "+" if (v["rp"][0] > 0 and v["t"][0] > 0) else ("-" if (v["rp"][1] < 0 and v["t"][1] < 0) else "0")
    vs, vsp = vonis(*v["sk"]), vonis(*v["sp"])
    hp = "" if padj is None else f" holm {padj:.3f}"
    r64 = ""
    if padj is not None:
        r64 = " R64:" + ("ya" if (vm == "+" and vs == "+" and padj < 0.05) else "tidak")
    print(f"{na:<22}{nr:<9}{v['m']:>+7.3f} [{v['rk'][0]:+.3f},{v['rk'][1]:+.3f}] t[{v['t'][0]:+.3f},{v['t'][1]:+.3f}] "
          f"prov[{v['rp'][0]:+.3f},{v['rp'][1]:+.3f}] p {v['p']:.3f}{hp} | satu {v['sr']:+.3f} "
          f"[{v['sk'][0]:+.3f},{v['sk'][1]:+.3f}] prov[{v['sp'][0]:+.3f},{v['sp'][1]:+.3f}] | "
          f"vonis rata {vm} prov {vpm} satu {vs} satu-prov {vsp}{r64}", flush=True)
    return vm, vpm, vs, vsp


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
    W = musim_wilayah(GID[g], m, o, d)
    mES, mLES = fit(h, W), fit(h, a, W)
    g, t, m = indeks(Y, [u])
    h, a = logit(rujukan(Y, tl, g, t, m)["ef1_halus"]), anomali(tl, t, m)
    o, d = iklim(t, m)
    W = musim_wilayah(GID[g], m, o, d)
    print(f"  {u}: EF-ES ONI/DMI [kem-SK, kem-lain, lain-SK, lain-lain] "
          + " ".join(f"{mES.coef_[0, 1 + 2 * k]:+.2f}/{mES.coef_[0, 2 + 2 * k]:+.2f}" for k in range(4))
          + f" | EF-LES nas {mLES.coef_[0, 1]:+.3f}", flush=True)
    print(f"  {u}: latih {len(yl):,} baris | EF-L b {mL.coef_[0, 0]:+.3f} nas {mL.coef_[0, 1]:+.3f} | "
          f"EF-E b {mE.coef_[0, 0]:+.3f} ONI {mE.coef_[0, 1]:+.3f} DMI {mE.coef_[0, 2]:+.3f} | EF-LE b {mLE.coef_[0, 0]:+.3f} "
          f"nas {mLE.coef_[0, 1]:+.3f} ONI {mLE.coef_[0, 2]:+.3f} DMI {mLE.coef_[0, 3]:+.3f}", flush=True)
    return pd.DataFrame({"tag": str(u), "gid": GID[g], "bulan": [f"{THN[p]}-{q + 1:02d}" for p, q in zip(t, m)],
                         "efl_t": mL.predict_proba(np.column_stack([h, a]))[:, 1],
                         "efe": mE.predict_proba(np.column_stack([h, o, d]))[:, 1],
                         "efle": mLE.predict_proba(np.column_stack([h, a, o, d]))[:, 1],
                         "efes": mES.predict_proba(np.column_stack([h, W]))[:, 1],
                         "efles": mLES.predict_proba(np.column_stack([h, a, W]))[:, 1]})


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
assert len(U) == n0 and not U[["efl_t", "efe", "efle", "efes", "efles"]].isna().any().any(), "baris uji tidak cocok dengan fase S"
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


KOL = [REF, "rujukan_peluang", "ef1_tingkat", "efld", "efe", "efle", "efes", "efles"] + MODEL
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
RP = dua_tahap_provinsi(S, grup_provinsi(kab), N, np.random.default_rng(B.SEED + 1))
NAMA = {REF: "kalender", "rujukan_peluang": "EF-P", "ef1_tingkat": "EF-L", "efld": "EF-LD", "efe": "EF-E", "efle": "EF-LE",
        "efes": "EF-ES", "efles": "EF-LES", "rf": "RF", "xgb": "GB", "xgb_setel": "GB tersetel", "mlp": "MLP",
        "xgb_nas": "GB + nasional", "xgb_nasdc": "GB + nasional + DC"}
NB = np.array([(g.tag == t).sum() for t in tags])
print(f"kabupaten {len(kab)}, provinsi {len(grup_provinsi(kab))}")
print("Brier rata-rata tahun: " + "  ".join(f"{NAMA[c]} {(th[:, ix[c]] / NB).mean():.5f}"
                                         for c in (REF, "ef1_tingkat", "efe", "efes", "efle", "efles", "efld")))
print("\n(a) rujukan")
for a, r in [("efes", REF), ("efes", "efe"), ("efes", "ef1_tingkat"), ("ef1_tingkat", "efes"), ("efles", "ef1_tingkat"),
             ("efles", "efes"), ("ef1_tingkat", "efe"), ("efe", REF)]:
    cetak(nilai(th, RB, RP, ix[a], ix[r]), NAMA[a], NAMA[r])
print("\n(b) model atas rujukan, Holm per keluarga (rujukan)")
for r in (REF, "rujukan_peluang", "ef1_tingkat", "efld", "efe", "efes"):
    V = [nilai(th, RB, RP, ix[a], ix[r]) for a in MODEL]
    ph = holm([v["p"] for v in V])
    print(f"-- atas {NAMA[r]}")
    for a, v, q in zip(MODEL, V, ph):
        cetak(v, NAMA[a], NAMA[r], q)
print("selesai.")
