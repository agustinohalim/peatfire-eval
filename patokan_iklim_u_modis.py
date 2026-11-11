"""
Revisi fase U (Hasil_Patokan_27), jalur MODIS: EF-ES dan EF-LES pada prakiraan maju 2008-2019 (lipatan R fase N),
satu rasio gabungan, klaster provinsi; satu sasaran per jalan. Bentuk EF-ES/EF-LES sama dengan patokan_iklim_u.py.
EF-L, EF-E dihitung ulang dengan kode fase T (patokan_iklim_t_modis.py); SSE EF-L dicek terhadap fase N. Kalender,
EF-P, EF-L, EF-LD dan GB-FH (rata-rata enam benih) dari DL_FIRE_MODIS/sse_modis_n_<nas|kab>.npz.

Pakai:
    python -u patokan_iklim_u_modis.py y_kabupaten > DL_FIRE_MODIS/patokan_iklim_u_modis_kab.log
    python -u patokan_iklim_u_modis.py y_gabungan  > DL_FIRE_MODIS/patokan_iklim_u_modis_nas.log
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
    W = musim_wilayah(GID[g], m, o, d)
    mES, mLES = fit(h, W), fit(h, a, W)
    g2, t2, m2 = indeks(Y, [u])
    h, a = logit(rujukan(Y, tl, g2, t2, m2)["ef1_halus"]), anomali(tl, t2, m2)
    o, d = iklim(t2, m2)
    W = musim_wilayah(GID[g2], m2, o, d)
    print(f"  {u}: EF-ES ONI/DMI [kem-SK, kem-lain, lain-SK, lain-lain] "
          + " ".join(f"{mES.coef_[0, 1 + 2 * k]:+.2f}/{mES.coef_[0, 2 + 2 * k]:+.2f}" for k in range(4))
          + f" | EF-LES nas {mLES.coef_[0, 1]:+.3f}", flush=True)
    print(f"  {u}: latih {len(tl)} thn {len(yl):,} baris | EF-L b {mL.coef_[0, 0]:+.3f} nas {mL.coef_[0, 1]:+.3f} | "
          f"EF-E b {mE.coef_[0, 0]:+.3f} ONI {mE.coef_[0, 1]:+.3f} DMI {mE.coef_[0, 2]:+.3f} | EF-LE nas "
          f"{mLE.coef_[0, 1]:+.3f} ONI {mLE.coef_[0, 2]:+.3f} DMI {mLE.coef_[0, 3]:+.3f}", flush=True)
    return pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2],
                         "efl_t": mL.predict_proba(np.column_stack([h, a]))[:, 1],
                         "efe": mE.predict_proba(np.column_stack([h, o, d]))[:, 1],
                         "efle": mLE.predict_proba(np.column_stack([h, a, o, d]))[:, 1],
                         "efes": mES.predict_proba(np.column_stack([h, W]))[:, 1],
                         "efles": mLES.predict_proba(np.column_stack([h, a, W]))[:, 1]})


uji = list(range(2008, 2020))
tags = [str(u) for u in uji]
P = pd.concat([lipatan(u) for u in uji], ignore_index=True)
kab = P.gid.unique(); pos = {k: i for i, k in enumerate(kab)}
BARU = ["efl_t", "efe", "efle", "efes", "efles"]
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
ok = {k: i for i, k in enumerate(LAMA + ["efe", "efle", "efes", "efles"])}
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
RP = dua_tahap_provinsi(S, grup_provinsi(list(kab)), N, np.random.default_rng(B.SEED + 1))
NAMA = {"kalender": "kalender", "efp": "EF-P", "efl": "EF-L", "efld": "EF-LD", "gb": "GB-FH", "efe": "EF-E", "efle": "EF-LE",
        "efes": "EF-ES", "efles": "EF-LES"}
print(f"\nU {SAS}, prakiraan maju 2008-2019, kabupaten {len(kab)}, provinsi {len(grup_provinsi(list(kab)))}")
print("\n(a) rujukan")
for a, r in [("efes", "kalender"), ("efes", "efe"), ("efes", "efl"), ("efl", "efes"), ("efles", "efl"), ("efles", "efes"),
             ("efl", "efe"), ("efe", "kalender")]:
    cetak(nilai(th, R, RP, ok[a], ok[r]), NAMA[a], NAMA[r])
print("\n(b) GB-FH atas rujukan (keluarga satu anggota, Holm = p)")
for r in ("kalender", "efp", "efl", "efld", "efe", "efes"):
    v = nilai(th, R, RP, ok["gb"], ok[r])
    cetak(v, "GB-FH", NAMA[r], v["p"])
print("selesai.")
