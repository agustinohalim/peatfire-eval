"""
Revisi fase Q (Hasil_Patokan_23), bagian MODIS maju 2008-2019: Q1 (tingkat yang diambil per tahun lawan galat
tingkat kalender, digabung dengan Q1 VIIRS dari patokan_tingkat_q.py) dan Q2 (dua tahap dengan klaster provinsi).
Bagian data dan fungsi disalin dari patokan_modis_o.py tanpa perubahan isi.

Pakai (sesudah patokan_tingkat_q.py untuk sasaran yang sama):
    python -u patokan_modis_q.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_q_kab.log
    python -u patokan_modis_q.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_q_nas.log
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
LANTAI = int(sys.argv[2]) if len(sys.argv) > 2 else None
KAPPA = 2.0
N = 1000
CALON = ["konstan", "konstan_bulan", "ef1", "ef3", "ef1_pasca", "ef1_halus"]
BENIH = [42, 1, 2, 3, 4, 5]
UNDIAN = 5
DM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_MODIS")
REF = "ef1" if SAS == "y_kabupaten" else "ef3"
RUJ = ["kalender", "efp", "efl"]
PASANGAN = [("gb", "efl"), ("efl", "kalender"), ("efl", "efp"), ("efp", "kalender"), ("gb", "kalender"), ("gb", "efp")]

V = pd.read_csv(os.path.join(B.D, "panel_nasional_bersih.csv"), usecols=["gid", "tahun", "titik_panas"])
M = pd.read_csv(os.path.join(DM, "panel_modis.csv"), usecols=["gid", "bulan", "tahun", "bulan_ke", "titik_panas"])
M = M[M.gid.isin(V.gid.unique()) & M.tahun.between(2003, 2025)].sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
vm = V[V.tahun.between(2012, 2025)].titik_panas.sum(); mm = M[M.tahun.between(2012, 2025)].titik_panas.sum()
r = mm / vm; F0 = max(1, int(round(10 * r)))
F = F0 if LANTAI is None else LANTAI
GID = np.sort(M.gid.unique()); THN = np.arange(2003, 2026)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
G_, T_, M_ = M.gid.map(gi).to_numpy(), M.tahun.map(ti).to_numpy(), (M.bulan_ke - 1).to_numpy()
TP = np.full((len(GID), len(THN), 12), np.nan)
TP[G_, T_, M_] = M.titik_panas.to_numpy(float)
NB = len(THN) * 12
TPf = TP.reshape(len(GID), NB)
del V, M
C = pd.concat([pd.read_csv(os.path.join(B.D, f), usecols=["GID_2", "bulan", "dc"]) for f in
               ("gfwed_komponen_MERRA2.CORRECTED_awal.csv", "gfwed_komponen_MERRA2.CORRECTED.csv")])
C = C.rename(columns={"GID_2": "gid"}).sort_values(["gid", "bulan"])
assert not C.duplicated(["gid", "bulan"]).any()
C["dc_lag1"] = C.groupby("gid").dc.shift(1)
C["tahun"] = C.bulan.str[:4].astype(int); C["_m"] = C.bulan.str[5:7].astype(int) - 1
C = C[C.gid.isin(GID) & C.tahun.between(2003, 2025)]
DC = np.full(TP.shape, np.nan)
DC[C.gid.map(gi).to_numpy(), C.tahun.map(ti).to_numpy(), C._m.to_numpy()] = C.dc_lag1.to_numpy()
hil = np.isnan(DC) & ~np.isnan(TP)
print(f"DC bulan lalu hilang di baris panel {hil.sum()} dari {(~np.isnan(TP)).sum()} (diisi median kabupaten); "
      f"kabupaten tanpa DC sama sekali {int(np.all(np.isnan(DC), (1, 2)).sum())}", flush=True)
med = np.nanmedian(DC.reshape(len(GID), -1), 1)
med = np.where(np.isnan(med), np.nanmedian(DC), med)
DC = np.log1p(np.where(np.isnan(DC), med[:, None, None], DC))
del C
print(f"MODIS/VIIRS 2012-2025 r {r:.3f}; F bawaan {F0}; lantai dipakai {F}; kabupaten {len(GID)}; sasaran {SAS}; "
      f"rujukan pasangan {REF}; benih {BENIH}", flush=True)

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


def X_gb(g, t, m, ef1, a, YL1):
    return np.column_stack([np.log1p(np.nan_to_num(LAG[k][g, t, m])) for k in (1, 2, 3, 12)]
                           + [SIN[g, t, m], COS[g, t, m], ef1, np.nan_to_num(YL1[g, t, m]), a])


UNDIAN_TL = {}
for u in range(2003, 2012):
    rg = np.random.default_rng(2026 + u)
    lain = np.array([x for x in THN if x != u])
    UNDIAN_TL[u] = [sorted(int(x) for x in rg.choice(lain, 12, replace=False)) for _ in range(UNDIAN)]


def lipatan(u, tl, ambang_latih, nama, gb=True):
    assert u not in tl
    L, ada = label(tl if ambang_latih else THN)
    Y = larik(L, ada)
    YL1 = jeda(L.reshape(len(GID), NB), 1).reshape(L.shape)
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
    dl = anomali_dc(tl, g, t, m, keluar_sendiri=True); hl = logit(rl["ef1_halus"])
    fit = lambda *X: LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack(X), yl)
    lg, mD, mLD = fit(hl, an), fit(hl, dl), fit(hl, an, dl)
    g2, t2, m2 = indeks(Y, [u])
    ru = rujukan(Y, tl, g2, t2, m2); au = anomali(tl, t2, m2); du = anomali_dc(tl, g2, t2, m2)
    hu = logit(ru["ef1_halus"])
    b = pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2], "kalender": ru[REF], "efp": ru[pilih],
                      "efl": lg.predict_proba(np.column_stack([hu, au]))[:, 1],
                      "efd": mD.predict_proba(np.column_stack([hu, du]))[:, 1],
                      "efld": mLD.predict_proba(np.column_stack([hu, au, du]))[:, 1]})
    print(f"    koef {u}: EF-L {lg.coef_[0, 0]:+.3f} {lg.coef_[0, 1]:+.3f} | EF-D {mD.coef_[0, 0]:+.3f} DC {mD.coef_[0, 1]:+.3f} | "
          f"EF-LD {mLD.coef_[0, 0]:+.3f} nas {mLD.coef_[0, 1]:+.3f} DC {mLD.coef_[0, 2]:+.3f}", flush=True)
    if gb:
        Xl = X_gb(g, t, m, rl["ef1"], an, YL1)
        Xu = X_gb(g2, t2, m2, ru["ef1"], au, YL1)
        for s in BENIH:
            xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                               reg_lambda=1.0, eval_metric="logloss", random_state=s, verbosity=0, n_jobs=-1)
            xg.fit(Xl, yl)
            b[f"gb_{s}"] = xg.predict_proba(Xu)[:, 1]
        Xl2, Xu2 = np.column_stack([Xl, dl]), np.column_stack([Xu, du])
        for s in BENIH:
            xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                               reg_lambda=1.0, eval_metric="logloss", random_state=s, verbosity=0, n_jobs=-1)
            xg.fit(Xl2, yl)
            b[f"gbdc_{s}"] = xg.predict_proba(Xu2)[:, 1]
    print(f"  {nama} {u}: latih {len(tl)} tahun, {len(yl):,} baris; uji {len(b):,}, prevalensi {b.y.mean() * 100:.2f}%; "
          f"EF-P {pilih}", flush=True)
    return b


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def matriks_sse(P, tags, rata):
    """S[tahun, kabupaten, keluaran]: jumlah galat kuadrat per kabupaten (rata-rata atas kolom keluaran)."""
    kab = P.gid.unique()
    pos = {k: i for i, k in enumerate(kab)}
    KEL = list(rata)
    S = np.zeros((len(tags), len(kab), len(KEL)))
    for a, t in enumerate(tags):
        x = P[P.tag == t]
        j = x.gid.map(pos).to_numpy()
        y = x.y.to_numpy()
        for c, k in enumerate(KEL):
            e = np.mean([(x[col].to_numpy() - y) ** 2 for col in rata[k]], axis=0)
            S[a, :, c] = np.bincount(j, weights=e, minlength=len(kab))
    return S, kab, {k: i for i, k in enumerate(KEL)}


def dua_tahap_cepat(S, n, rng):
    """Sama dengan B.dua_tahap (urutan panggilan rng sama): tiap ulangan satu tarikan kabupaten untuk semua
    tahun, lalu tahun diambil ulang. Hasil (n, tahun, keluaran)."""
    nt, nk, _ = S.shape
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        pilih = rng.choice(nk, nk)
        tt = rng.integers(0, nt, nt)
        cs = S[:, pilih, :].sum(1)
        out[r] = cs[tt]
    return out


def blok_bergerak(S, n, rng, panjang=3):
    """Kabupaten diambil ulang seperti dua tahap; tahun diambil sebagai blok berurutan melingkar."""
    nt, nk, _ = S.shape
    nb = int(np.ceil(nt / panjang))
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        pilih = rng.choice(nk, nk)
        mulai = rng.integers(0, nt, nb)
        tt = np.concatenate([(m0 + np.arange(panjang)) % nt for m0 in mulai])[:nt]
        out[r] = S[:, pilih, :].sum(1)[tt]
    return out


def ringkas(th, R, ok, a, r):
    pt = 1 - th[:, ok[a]] / th[:, ok[r]]
    rr = (1 - R[:, :, ok[a]] / R[:, :, ok[r]]).mean(1)
    sr = 1 - R[:, :, ok[a]].sum(1) / R[:, :, ok[r]].sum(1)
    return pt, np.percentile(rr, [2.5, 97.5]), np.percentile(sr, [2.5, 97.5])




def geser(f, ybar):
    lf = logit(f, 1e-6)
    h = lambda dl: np.mean(1 / (1 + np.exp(-(lf + dl)))) - ybar
    d = brentq(h, -30, 30)
    return 1 / (1 + np.exp(-(lf + d))), d




# ---------------------------------------------------------------- fase Q (Hasil_Patokan_23)
def grup_provinsi(kab):
    pv = pd.read_csv(f"{B.D}/panel_nasional_bersih.csv", usecols=["gid", "provinsi"]).drop_duplicates()
    peta = dict(zip(pv.gid, pv.provinsi))
    nama = sorted({peta[k] for k in kab})
    return [np.array([k for k in kab if peta[k] == p]) for p in nama]


def logit_q(p, e=1e-6):
    p = min(max(p, e), 1 - e)
    return np.log(p / (1 - p))


B.cek_ap()
t0 = time.time()
uji = list(range(2008, 2020))
tags = [str(u) for u in uji]
print("\n" + "=" * 100 + f"\nQ MODIS maju {SAS}  lantai {F}  (ambang dari tahun latih), n = 12\n" + "=" * 100, flush=True)
P = pd.concat([lipatan(u, [x for x in THN if x < u], True, "R") for u in uji], ignore_index=True)
rata = {c: [c] for c in ["kalender", "efp", "efl", "efd", "efld"]}
rata["gb"] = [f"gb_{s}" for s in BENIH]
rata["gb_dc"] = [f"gbdc_{s}" for s in BENIH]
# Q1: geser per tahun, tiap benih dan kalender sendiri
for t in tags:
    j = P.tag == t
    yb = P.loc[j, "y"].mean()
    for c in ["kalender"] + rata["gb"]:
        P.loc[j, c + "_lv"] = geser(P.loc[j, c].to_numpy(), yb)[0]
rata["kalender_lv"] = ["kalender_lv"]
rata["gb_lv"] = [f"gb_{s}_lv" for s in BENIH]
S, kab, ok = matriks_sse(P, tags, rata)
laju = P.groupby("tag").y.mean().reindex(tags).to_numpy()
kal = P.groupby("tag").kalender.mean().reindex(tags).to_numpy()
del P
th = S.sum(1)
print(f"  ({time.time() - t0:.0f} s) kabupaten {len(kab)}", flush=True)

print("\nQ1 MODIS maju: L = BSS mentah - BSS sesudah geser per tahun (GB-FH atas kalender); d = |logit kalender - logit teramati|")
q1 = []
for i, t in enumerate(tags):
    raw = 1 - th[i, ok["gb"]] / th[i, ok["kalender"]]
    lv = 1 - th[i, ok["gb_lv"]] / th[i, ok["kalender_lv"]]
    tanda = logit_q(kal[i]) - logit_q(laju[i])
    q1.append((t, raw, lv, raw - lv, abs(tanda), tanda, laju[i], kal[i]))
    print(f"  {t}: mentah {raw:+.3f}  geser {lv:+.3f}  L {raw - lv:+.3f}  d {abs(tanda):.3f} (tanda {tanda:+.3f}; "
          f"laju {laju[i] * 100:.2f}%, kalender {kal[i] * 100:.2f}%)", flush=True)
Q = pd.DataFrame(q1, columns=["tahun", "bss_mentah", "bss_geser", "L", "d", "tanda", "laju", "kalender"]).assign(rancangan="MODIS")
V = pd.read_csv(f"{B.D}/patokan_q1_viirs_{SAS}.csv")
G = pd.concat([V, Q], ignore_index=True)


def kemiringan(G):
    X = np.column_stack([G.d.to_numpy(), (G.rancangan == "VIIRS").to_numpy(float), (G.rancangan == "MODIS").to_numpy(float)])
    return np.linalg.lstsq(X, G.L.to_numpy(), rcond=None)[0][0]


b0 = kemiringan(G)
rng = np.random.default_rng(B.SEED)
gv, gm = G[G.rancangan == "VIIRS"].reset_index(drop=True), G[G.rancangan == "MODIS"].reset_index(drop=True)
bb = np.array([kemiringan(pd.concat([gv.iloc[rng.integers(0, len(gv), len(gv))], gm.iloc[rng.integers(0, len(gm), len(gm))]]))
               for _ in range(10000)])
lo, hi = np.percentile(bb, [2.5, 97.5])
print(f"  gabungan 18 tahun: kemiringan L atas d (intersep per rancangan) {b0:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
      f"R50 {'kemiringan > 0, selang tanpa nol' if lo > 0 else 'selang memuat nol atau negatif'}")
for nm, x in (("VIIRS", gv), ("MODIS", gm), ("gabungan", G)):
    r_, p_ = stats.spearmanr(x.d, x.L)
    print(f"  Spearman {nm:<9} rho {r_:+.3f} p {p_:.3f} (n {len(x)})")
r_, p_ = stats.spearmanr(G.tanda, G.L)
print(f"  deskriptif: Spearman L lawan tanda (kalender di atas teramati = positif) rho {r_:+.3f} p {p_:.3f}")

print(f"\nQ2 MODIS maju: dua tahap, kabupaten lawan klaster provinsi (1.000 ulangan, benih {B.SEED})")
grup_n = grup_provinsi(kab)
pos = {k: i for i, k in enumerate(kab)}
grup = [np.array([pos[k] for k in gk]) for gk in grup_n]
print(f"  provinsi {len(grup)}; kabupaten {len(kab)}")


def dua_tahap_prov(S, n, rng):
    nt = S.shape[0]
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        pr = rng.integers(0, len(grup), len(grup))
        pilih = np.concatenate([grup[i] for i in pr])
        tt = rng.integers(0, nt, nt)
        out[r] = S[:, pilih, :].sum(1)[tt]
    return out


R = dua_tahap_cepat(S, N, np.random.default_rng(B.SEED))
RP = dua_tahap_prov(S, N, np.random.default_rng(B.SEED))
for a, r in (("gb", "kalender"), ("gb", "efp"), ("gb", "efl"), ("gb", "efld"), ("gb_dc", "kalender"), ("gb_dc", "efp"),
             ("gb_dc", "efl"), ("gb_dc", "efld"), ("efl", "kalender"), ("efld", "efl")):
    pt, qk, _ = ringkas(th, R, ok, a, r)
    _, qp, _ = ringkas(th, RP, ok, a, r)
    tl, tu = t95(pt)
    teks = ""
    if abs(qp[0]) <= 0.01:
        _, q10, _ = ringkas(th, dua_tahap_prov(S, 10000, np.random.default_rng(B.SEED)), ok, a, r)
        teks = f"  10.000 ulangan provinsi [{q10[0]:+.4f}, {q10[1]:+.4f}]"
    print(f"    {a}/{r:<9} {pt.mean():+.4f}  kabupaten [{qk[0]:+.4f}, {qk[1]:+.4f}]  provinsi [{qp[0]:+.4f}, {qp[1]:+.4f}]  "
          f"tahun t [{tl:+.4f}, {tu:+.4f}]{teks}", flush=True)
print("selesai.")
