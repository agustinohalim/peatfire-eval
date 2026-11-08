"""
Revisi fase K (Hasil_Patokan_16): agregasi prakiraan maju MODIS sesudah panel 12.
  K1  R  prakiraan maju 2008-2019 (sama persis dengan fase J): BSS per pasangan dengan lima agregasi
         (a) rata-rata rasio tahunan (= fase J), dua tahap + t95
         (b) satu rasio atas semua baris uji (1 - jumlah galat kuadrat model / jumlah rujukan), dua tahap
         (c) median rasio tahunan, banyak tahun > 0, uji tanda dua sisi
         (d) rata-rata rasio tanpa tiga tahun uji berprevalensi terendah, dua tahap + t95 (df 8)
         (e) deskriptif: rata-rata 2008-2015 dan 2016-2019 (tahun uji pertama dengan latih sesudah 2015)
  K2  pasangan baru EF-L atas EF-P (memisahkan penyesuaian tingkat dari penghalusan kalender)
  K3  Brier rujukan V3 (12 tahun acak, lima undian) atas V1 (latih < 2016) pada baris uji yang sama
      (kalender, EF-P, EF-L; tanpa GB), dua tahap + t95 (df 8)
  K4  lantai kejadian per kabupaten: argumen kedua = lantai (default F = bulat(10 r) = 2); K1/K2 diulang
      dengan lantai 3 dan 5 (hanya y_kabupaten, tanpa K3)
Panel, fitur, parameter, benih, EF-P dan EF-L sama dengan patokan_modis_j.py.

Pakai:
    python -u patokan_modis_k.py y_kabupaten   > DL_FIRE_MODIS/patokan_modis_k_kab.log
    python -u patokan_modis_k.py y_gabungan    > DL_FIRE_MODIS/patokan_modis_k_nas.log
    python -u patokan_modis_k.py y_kabupaten 3 > DL_FIRE_MODIS/patokan_modis_k_kab_l3.log
    python -u patokan_modis_k.py y_kabupaten 5 > DL_FIRE_MODIS/patokan_modis_k_kab_l5.log
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


def mdd80(pt):
    n = len(pt)
    return (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * pt.std(ddof=1) / np.sqrt(n)


def sse_tahunan(P, tags, rata):
    """Jumlah galat kuadrat per tahun dan per keluaran (rata-rata atas kolom), titik dan dua tahap."""
    KOL = [c for c in P.columns if c not in ("gid", "tag", "y")]
    KEL = list(rata)

    def fn(bk, j):
        y = bk.y[j]
        s = {c: np.sum((bk.s[c][j] - y) ** 2) for c in KOL}
        return np.array([np.mean([s[c] for c in rata[k]]) for k in KEL])

    kab = P.gid.unique()
    blok = {t: B.Blok(x, KOL, kab) for t, x in P.groupby("tag")}
    blok = {t: blok[t] for t in tags}
    th = np.array([fn(blok[t], slice(None)) for t in tags])
    R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)
    return th, R2, {c: i for i, c in enumerate(KEL)}


def rata_biasa():
    d = {c: [c] for c in RUJ}
    d["gb"] = [f"gb_{s}" for s in BENIH]
    return d


def agregasi(P, uji, judul):
    tags = [str(u) for u in uji]
    prev = P.groupby("tag").y.mean().reindex(tags).to_numpy()
    sepi = [tags[i] for i in np.argsort(prev)[:3]]
    print(f"  {judul}; tiga tahun tersepi: " + ", ".join(f"{t} ({prev[tags.index(t)] * 100:.2f}%)" for t in sepi), flush=True)
    th, R2, ok = sse_tahunan(P, tags, rata_biasa())
    sisa = [t for t in tags if t not in sepi]
    thS, R2S, _ = sse_tahunan(P[P.tag.isin(sisa)], sisa, rata_biasa())
    awal = [t for t in tags if int(t) <= 2015]; akhir = [t for t in tags if int(t) >= 2016]
    hasil = {}
    for a, rr in PASANGAN:
        ia, ir = ok[a], ok[rr]
        pt = 1 - th[:, ia] / th[:, ir]; dist = (1 - R2[:, :, ia] / R2[:, :, ir]).mean(1)
        lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
        sr = 1 - th[:, ia].sum() / th[:, ir].sum(); dsr = 1 - R2[:, :, ia].sum(1) / R2[:, :, ir].sum(1)
        slo, shi = np.nanpercentile(dsr, [2.5, 97.5])
        med = np.median(pt); npos = int((pt > 0).sum())
        ptanda = stats.binomtest(npos, len(pt), 0.5).pvalue
        ps = 1 - thS[:, ia] / thS[:, ir]; dS = (1 - R2S[:, :, ia] / R2S[:, :, ir]).mean(1)
        plo, phi = np.nanpercentile(dS, [2.5, 97.5]); stl, stu = t95(ps)
        pa = np.array([pt[tags.index(t)] for t in awal]); pk = np.array([pt[tags.index(t)] for t in akhir])
        print(f"\n    {a} atas {rr}", flush=True)
        print(f"      (a) rata-rata rasio  {pt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}] "
              f"MDD80 {mdd80(pt):.4f}{'  LOLOS' if lo > 0 and tl > 0 else ''}", flush=True)
        print(f"      (b) satu rasio       {sr:+.4f} 2tahap [{slo:+.4f}, {shi:+.4f}]{'  LOLOS' if slo > 0 else ''}"
              f"{'  TANDA BERBALIK' if np.sign(sr) != np.sign(pt.mean()) else ''}", flush=True)
        print(f"      (c) median tahunan   {med:+.4f}; tahun > 0: {npos} dari {len(pt)}; uji tanda p = {ptanda:.3f}", flush=True)
        print(f"      (d) tanpa 3 tersepi  {ps.mean():+.4f} 2tahap [{plo:+.4f}, {phi:+.4f}] t95 [{stl:+.4f}, {stu:+.4f}] "
              f"MDD80 {mdd80(ps):.4f}{'  LOLOS' if plo > 0 and stl > 0 else ''}", flush=True)
        print(f"      (e) 2008-2015 {pa.mean():+.4f} (t95 [{t95(pa)[0]:+.4f}, {t95(pa)[1]:+.4f}]); "
              f"2016-2019 {pk.mean():+.4f} (t95 [{t95(pk)[0]:+.4f}, {t95(pk)[1]:+.4f}])", flush=True)
        print("      per tahun " + " ".join(f"{t}:{x:+.3f}" for t, x in zip(tags, pt)), flush=True)
        hasil[(a, rr)] = dict(rr=pt.mean(), lolos_rr=lo > 0 and tl > 0, sr=sr, slo=slo, shi=shi, med=med,
                              npos=npos, ps=ps.mean(), lolos_ps=plo > 0 and stl > 0)
    return hasil


B.cek_ap()
t0 = time.time()
uji = list(range(2008, 2020))
print("\n" + "=" * 100 + f"\nK R  {SAS}  lantai {F}  (prakiraan maju 2008-2019, ambang dari tahun latih), n = 12\n"
      + "=" * 100, flush=True)
P = pd.concat([lipatan(u, [x for x in THN if x < u], True, "R") for u in uji], ignore_index=True)
H = agregasi(P, uji, f"({time.time() - t0:.0f} s)")
del P
print("\nR39 ringkas: " + "; ".join(
    f"{a}/{rr} rr {v['rr']:+.3f}{' L' if v['lolos_rr'] else ''} satu {v['sr']:+.3f} [{v['slo']:+.3f}, {v['shi']:+.3f}] "
    f"med {v['med']:+.3f} ({v['npos']}/12) tanpa-sepi {v['ps']:+.3f}{' L' if v['lolos_ps'] else ''}"
    for (a, rr), v in H.items()), flush=True)

if LANTAI is None:
    # ------------------------------------------------------------- K3: Brier rujukan V3 atas V1
    t0 = time.time()
    uji = list(range(2003, 2012))
    print("\n" + "=" * 100 + f"\nK D  {SAS}  (Brier rujukan V3 12 tahun acak atas V1 latih < 2016, tanpa GB), n = 9\n"
          + "=" * 100, flush=True)
    bag = []
    for u in uji:
        fr = [lipatan(u, [x for x in THN if x != u and x < 2016], False, "V1", gb=False)]
        fr += [lipatan(u, UNDIAN_TL[u][d], False, f"V3 undian {d}", gb=False) for d in range(UNDIAN)]
        for k in range(1, len(fr)):
            assert np.array_equal(fr[k].y.to_numpy(), fr[0].y.to_numpy())
        b = fr[0][["gid", "tag", "y"]].copy()
        for c in RUJ:
            b[f"V1_{c}"] = fr[0][c].to_numpy()
            for d in range(UNDIAN):
                b[f"V3_{d}_{c}"] = fr[d + 1][c].to_numpy()
        bag.append(b)
    P = pd.concat(bag, ignore_index=True)
    rata = {f"V1_{c}": [f"V1_{c}"] for c in RUJ}
    rata.update({f"V3_{c}": [f"V3_{d}_{c}" for d in range(UNDIAN)] for c in RUJ})
    tags = [str(u) for u in uji]
    th, R2, ok = sse_tahunan(P, tags, rata)
    print(f"  ({time.time() - t0:.0f} s) BSS Brier rujukan V3 atas V1 (positif = rujukan 12 tahun acak lebih baik)")
    for c in RUJ:
        pt = 1 - th[:, ok[f"V3_{c}"]] / th[:, ok[f"V1_{c}"]]
        dist = (1 - R2[:, :, ok[f"V3_{c}"]] / R2[:, :, ok[f"V1_{c}"]]).mean(1)
        lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
        tanda = "LEBIH BURUK" if hi < 0 and tu < 0 else ("LEBIH BAIK" if lo > 0 and tl > 0 else "tak terbedakan")
        print(f"    {c:<9} {pt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}] MDD80 {mdd80(pt):.4f}"
              f"  {tanda}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
print("selesai.")
