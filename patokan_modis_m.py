"""
Revisi fase M (Hasil_Patokan_18), M2a/M2b: galat Monte Carlo bootstrap dan bootstrap blok bergerak atas tahun
untuk prakiraan maju MODIS 2008-2019 (lipatan R sama persis dengan fase J/K: latih = tahun MODIS sebelum tahun
uji, ambang dan kabupaten dari tahun latih, GB enam benih).
  M2a dua tahap (kabupaten lalu tahun) dengan lima benih x 1.000 ulangan dan 10.000 ulangan benih B.SEED
  M2b kabupaten diambil ulang, tahun uji sebagai blok berurutan sepanjang 3 (melingkar), 10.000 ulangan
GB atas EF-L, kalender, EF-P dan EF-L atas kalender; rata-rata rasio tahunan dan satu rasio atas semua baris.
Galat kuadrat per tahun per kabupaten disimpan ke DL_FIRE_MODIS/sse_modis_m_{nas,kab}.npz (tidak masuk repo).

Pakai:
    python -u patokan_modis_m.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_m_nas.log
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
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
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack([logit(rl["ef1_halus"]), an]), yl)
    g2, t2, m2 = indeks(Y, [u])
    ru = rujukan(Y, tl, g2, t2, m2); au = anomali(tl, t2, m2)
    b = pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2], "kalender": ru[REF], "efp": ru[pilih],
                      "efl": lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), au]))[:, 1]})
    if gb:
        Xl = X_gb(g, t, m, rl["ef1"], an, YL1)
        Xu = X_gb(g2, t2, m2, ru["ef1"], au, YL1)
        for s in BENIH:
            xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                               reg_lambda=1.0, eval_metric="logloss", random_state=s, verbosity=0, n_jobs=-1)
            xg.fit(Xl, yl)
            b[f"gb_{s}"] = xg.predict_proba(Xu)[:, 1]
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


B.cek_ap()
t0 = time.time()
uji = list(range(2008, 2020))
tags = [str(u) for u in uji]
print("\n" + "=" * 100 + f"\nM2 R  {SAS}  lantai {F}  (prakiraan maju 2008-2019, ambang dari tahun latih), n = 12\n"
      + "=" * 100, flush=True)
P = pd.concat([lipatan(u, [x for x in THN if x < u], True, "R") for u in uji], ignore_index=True)
rata = {c: [c] for c in RUJ}
rata["gb"] = [f"gb_{s}" for s in BENIH]
S, kab, ok = matriks_sse(P, tags, rata)
prev = P.groupby("tag").y.mean().reindex(tags).to_numpy()
np.savez_compressed(os.path.join(DM, f"sse_modis_m_{'nas' if SAS == 'y_gabungan' else 'kab'}.npz"),
                    S=S, kab=kab.astype(str), keluaran=np.array(list(ok)), tags=np.array(tags), prev=prev)
del P
th = S.sum(1)
print(f"  ({time.time() - t0:.0f} s) kabupaten {len(kab)}; prevalensi per tahun " +
      " ".join(f"{t}:{p * 100:.2f}%" for t, p in zip(tags, prev)), flush=True)

PAS = [("gb", "efl"), ("gb", "kalender"), ("gb", "efp"), ("efl", "kalender")]
print("\nM2a galat Monte Carlo: dua tahap, lima benih x 1.000 dan 10.000 ulangan benih B.SEED")
print("    (rata-rata rasio [2 tahap] t95 | satu rasio [2 tahap])")
for a, r in PAS:
    pt = 1 - th[:, ok[a]] / th[:, ok[r]]
    tl, tu = t95(pt)
    sr = 1 - th[:, ok[a]].sum() / th[:, ok[r]].sum()
    print(f"  {a}/{r}: rata-rata rasio {pt.mean():+.4f} t95 [{tl:+.4f}, {tu:+.4f}]  satu rasio {sr:+.4f}", flush=True)
HAS = {}
for nama, n, benih in [(f"benih {b}", 1000, b) for b in (B.SEED, 1, 2, 3, 4)] + [("10.000 ulangan", 10000, B.SEED)]:
    R = dua_tahap_cepat(S, n, np.random.default_rng(benih))
    for a, r in PAS:
        pt, q1, q2 = ringkas(th, R, ok, a, r)
        HAS.setdefault((a, r), []).append((nama, q1, q2))
        print(f"    {nama:<15} {a}/{r:<9} rata-rata rasio [{q1[0]:+.4f}, {q1[1]:+.4f}]  satu rasio [{q2[0]:+.4f}, {q2[1]:+.4f}]",
              flush=True)
print("\n  rentang batas bawah atas enam jalan:")
for (a, r), v in HAS.items():
    lo1 = [x[1][0] for x in v]; lo2 = [x[2][0] for x in v]
    print(f"    {a}/{r:<9} rata-rata rasio {min(lo1):+.4f} .. {max(lo1):+.4f}  semua > 0: {all(x > 0 for x in lo1)}   "
          f"satu rasio {min(lo2):+.4f} .. {max(lo2):+.4f}  semua > 0: {all(x > 0 for x in lo2)}", flush=True)

print("\nM2b bootstrap blok bergerak atas tahun uji (panjang 3, melingkar), kabupaten diambil ulang, 10.000 ulangan")
R = blok_bergerak(S, 10000, np.random.default_rng(B.SEED))
for a, r in PAS:
    pt, q1, q2 = ringkas(th, R, ok, a, r)
    print(f"    {a}/{r:<9} rata-rata rasio {pt.mean():+.4f} [{q1[0]:+.4f}, {q1[1]:+.4f}]  satu rasio [{q2[0]:+.4f}, {q2[1]:+.4f}]",
          flush=True)
print("\n  per tahun (untuk M3): " + "; ".join(
    f"{a}/{r} " + " ".join(f"{t}:{x:+.3f}" for t, x in zip(tags, 1 - th[:, ok[a]] / th[:, ok[r]])) for a, r in PAS[:1] + PAS[3:]))
print("selesai.")
