"""
Revisi fase J (Hasil_Patokan_15): jalur MODIS sesudah panel 11.
  J1  R   asal bergulir: uji 2008-2019 (12 tahun, df 11), latih = semua tahun MODIS SEBELUM tahun uji,
          ambang dan kabupaten yang dinilai dari tahun latih (prakiraan sungguhan; tak ada tahun sesudah uji)
  J1b V5  V0 dengan penyangga jeda: latih = semua tahun lain kecuali u+1 (baris u+1 memuat jeda dari u)
      V6  V2 dengan penyangga jeda: seperti V5, ambang dan kabupaten dari tahun latih
  J2  D   beda langsung V3 (12 tahun acak, lima undian) - V1 (latih < 2016), baris uji sama (ambang 2003-2025),
          BSS GB atas EF-L / kalender / EF-P per tahun, dua tahap bersama dan t95 (df 8)
Panel, fitur, parameter, benih, EF-P dan EF-L sama dengan patokan_modis_i.py (fase I).
GB enam benih {42, 1, 2, 3, 4, 5}; Brier GB per tahun = rata-rata atas benih (dan undian).

Pakai (satu sasaran per jalan):
    python -u patokan_modis_j.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_j_kab.log
    python -u patokan_modis_j.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_j_nas.log
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
KAPPA = 2.0
N = 1000
CALON = ["konstan", "konstan_bulan", "ef1", "ef3", "ef1_pasca", "ef1_halus"]
BENIH = [42, 1, 2, 3, 4, 5]
UNDIAN = 5
DM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_MODIS")
REF = "ef1" if SAS == "y_kabupaten" else "ef3"
RUJ = ["kalender", "efp", "efl"]

V = pd.read_csv(os.path.join(B.D, "panel_nasional_bersih.csv"), usecols=["gid", "tahun", "titik_panas"])
M = pd.read_csv(os.path.join(DM, "panel_modis.csv"), usecols=["gid", "bulan", "tahun", "bulan_ke", "titik_panas"])
M = M[M.gid.isin(V.gid.unique()) & M.tahun.between(2003, 2025)].sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
vm = V[V.tahun.between(2012, 2025)].titik_panas.sum(); mm = M[M.tahun.between(2012, 2025)].titik_panas.sum()
r = mm / vm; F = max(1, int(round(10 * r)))
GID = np.sort(M.gid.unique()); THN = np.arange(2003, 2026)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
G_, T_, M_ = M.gid.map(gi).to_numpy(), M.tahun.map(ti).to_numpy(), (M.bulan_ke - 1).to_numpy()
TP = np.full((len(GID), len(THN), 12), np.nan)
TP[G_, T_, M_] = M.titik_panas.to_numpy(float)
NB = len(THN) * 12
TPf = TP.reshape(len(GID), NB)
del V, M
print(f"MODIS/VIIRS 2012-2025 r {r:.3f}; F = {F}; kabupaten {len(GID)}; sasaran {SAS}; rujukan pasangan {REF}; "
      f"benih {BENIH}", flush=True)

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


# undian V3 sama persis dengan fase I: lima undian berurutan dari default_rng(2026 + u)
UNDIAN_TL = {}
for u in range(2003, 2012):
    rg = np.random.default_rng(2026 + u)
    lain = np.array([x for x in THN if x != u])
    UNDIAN_TL[u] = [sorted(int(x) for x in rg.choice(lain, 12, replace=False)) for _ in range(UNDIAN)]


def lipatan(u, tl, ambang_latih, nama):
    """Satu lipatan: tahun uji u, tahun latih tl; ambang dari tl bila ambang_latih, kalau tidak dari 2003-2025."""
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
    Xl = X_gb(g, t, m, rl["ef1"], an, YL1)
    g2, t2, m2 = indeks(Y, [u])
    ru = rujukan(Y, tl, g2, t2, m2); au = anomali(tl, t2, m2)
    Xu = X_gb(g2, t2, m2, ru["ef1"], au, YL1)
    b = pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2], "kalender": ru[REF], "efp": ru[pilih],
                      "efl": lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), au]))[:, 1]})
    for s in BENIH:
        xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                           reg_lambda=1.0, eval_metric="logloss", random_state=s, verbosity=0, n_jobs=-1)
        xg.fit(Xl, yl)
        b[f"gb_{s}"] = xg.predict_proba(Xu)[:, 1]
    print(f"  {nama} {u}: latih {len(tl)} tahun ({','.join(str(x)[2:] for x in tl)}), {len(yl):,} baris; "
          f"uji {len(b):,}, prevalensi {b.y.mean() * 100:.2f}%; EF-P {pilih}; koef EF-L {lg.coef_[0][1]:+.2f}", flush=True)
    return b


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def mdd80(pt):
    n = len(pt)
    return (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * pt.std(ddof=1) / np.sqrt(n)


def cetak(nama, pt, dist):
    lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
    lolos = lo > 0 and tl > 0
    print(f"    {nama:<26}{pt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}] MDD80 {mdd80(pt):.4f}"
          f"{'  LOLOS' if lolos else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
    return lolos


def gabung(daftar, awalan):
    """Gabungkan beberapa keluaran lipatan (tahun uji sama) menjadi satu tabel dengan kolom berawalan."""
    for k in range(1, len(daftar)):
        assert np.array_equal(daftar[k].y.to_numpy(), daftar[0].y.to_numpy())
        assert np.array_equal(daftar[k].gid.to_numpy(), daftar[0].gid.to_numpy())
    b = daftar[0][["gid", "tag", "y"]].copy()
    for x, a in zip(daftar, awalan):
        for c in x.columns[3:]:
            b[f"{a}{c}"] = x[c].to_numpy()
    return b


def analisis(P, tags, rata, judul):
    """rata: nama keluaran -> daftar kolom yang Brier-nya dirata-rata. Mengembalikan (th, R2, ok)."""
    KOL = [c for c in P.columns if c not in ("gid", "tag", "y")]
    KEL = list(rata)
    ok = {c: i for i, c in enumerate(KEL)}

    def fn(bk, j):
        y = bk.y[j]
        bs = {c: np.mean((bk.s[c][j] - y) ** 2) for c in KOL}
        return np.array([np.mean([bs[c] for c in rata[k]]) for k in KEL])

    kab = P.gid.unique()
    blok = {t: B.Blok(x, KOL, kab) for t, x in P.groupby("tag")}
    th = np.array([fn(blok[t], slice(None)) for t in tags])
    R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)
    print(f"  {judul}: Brier rata-rata " + "  ".join(f"{c} {th[:, ok[c]].mean():.5f}" for c in KEL if "_" not in c), flush=True)
    return th, R2, ok


def lapor(th, R2, ok, nama, uji):
    bss = lambda a, rr: (1 - th[:, ok[a]] / th[:, ok[rr]], (1 - R2[:, :, ok[a]] / R2[:, :, ok[rr]]).mean(1))
    print(f"  BSS (gb = rata-rata enam benih), n = {len(uji)}, df {len(uji) - 1}")
    lolos = {}
    for a, rr in (("efl", "kalender"), ("efp", "kalender"), ("gb", "kalender"), ("gb", "efp"), ("gb", "efl")):
        lolos[(a, rr)] = cetak(f"{a} atas {rr}", *bss(a, rr))
    semua = all(lolos[("gb", x)] for x in RUJ)
    gagal = [x for x in RUJ if not lolos[("gb", x)]]
    nlolos = 0
    for s in BENIH:
        ls = []
        for rr in RUJ:
            pt, ds = bss(f"gb{s}", rr)
            ls.append(np.nanpercentile(ds, 2.5) > 0 and t95(pt)[0] > 0)
        nlolos += all(ls)
    for rr in RUJ:
        v = [bss(f"gb{s}", rr)[0].mean() for s in BENIH]
        print(f"    rentang benih gb atas {rr:<9} {min(v):+.4f} .. {max(v):+.4f}")
    print(f"  R38 {nama} rata-rata benih: kalender {lolos[('gb', 'kalender')]}, EF-P {lolos[('gb', 'efp')]}, "
          f"EF-L {lolos[('gb', 'efl')]} -> mengalahkan ketiga rujukan: {semua}"
          + (f" (tidak: {', '.join(gagal)})" if gagal else "") + f"; benih lolos sendiri {nlolos} dari 6", flush=True)
    return semua, gagal, nlolos, lolos[("gb", "efl")]


def rata_biasa(awal=""):
    d = {c: [f"{awal}{c}"] for c in RUJ}
    d["gb"] = [f"{awal}gb_{s}" for s in BENIH]
    d.update({f"gb{s}": [f"{awal}gb_{s}"] for s in BENIH})
    return d


B.cek_ap()
RINGKAS = {}
# ------------------------------------------------------------- J1 dan J1b
DESAIN = [
    ("R", list(range(2008, 2020)), lambda u: [x for x in THN if x < u], True,
     "asal bergulir: latih tahun sebelum uji, ambang dari tahun latih"),
    ("V5", list(range(2003, 2012)), lambda u: [x for x in THN if x not in (u, u + 1)], False,
     "V0 + penyangga jeda (tanpa u+1), ambang 2003-2025"),
    ("V6", list(range(2003, 2012)), lambda u: [x for x in THN if x not in (u, u + 1)], True,
     "V2 + penyangga jeda (tanpa u+1), ambang dari tahun latih"),
]
for nama, uji, latih, ambang_latih, ket in DESAIN:
    t0 = time.time()
    print("\n" + "=" * 100 + f"\nJ {nama}  {SAS}  ({ket}), n = {len(uji)}\n" + "=" * 100, flush=True)
    P = pd.concat([lipatan(u, latih(u), ambang_latih, nama) for u in uji], ignore_index=True)
    th, R2, ok = analisis(P, [str(u) for u in uji], rata_biasa(), f"({time.time() - t0:.0f} s)")
    RINGKAS[nama] = lapor(th, R2, ok, nama, uji)
    del P, R2

# ------------------------------------------------------------- J2: beda V3 - V1 pada baris uji yang sama
t0 = time.time()
uji = list(range(2003, 2012))
print("\n" + "=" * 100 + f"\nJ D  {SAS}  (beda langsung V3 12 tahun acak - V1 latih < 2016), n = 9\n" + "=" * 100, flush=True)
bag = []
for u in uji:
    fr = [lipatan(u, [x for x in THN if x != u and x < 2016], False, "V1")]
    fr += [lipatan(u, UNDIAN_TL[u][d], False, f"V3 undian {d}") for d in range(UNDIAN)]
    bag.append(gabung(fr, ["a_"] + [f"c{d}_" for d in range(UNDIAN)]))
P = pd.concat(bag, ignore_index=True)
rata = {f"V1_{c}": [f"a_{c}"] for c in RUJ}
rata["V1_gb"] = [f"a_gb_{s}" for s in BENIH]
rata.update({f"V3_{c}": [f"c{d}_{c}" for d in range(UNDIAN)] for c in RUJ})
rata["V3_gb"] = [f"c{d}_gb_{s}" for s in BENIH for d in range(UNDIAN)]
th, R2, ok = analisis(P, [str(u) for u in uji], rata, f"({time.time() - t0:.0f} s)")
print("  BSS per desain dan beda V3 - V1 (positif = 12 tahun acak lebih baik)")
for rr in RUJ:
    p3 = 1 - th[:, ok["V3_gb"]] / th[:, ok[f"V3_{rr}"]]; d3 = (1 - R2[:, :, ok["V3_gb"]] / R2[:, :, ok[f"V3_{rr}"]]).mean(1)
    p1 = 1 - th[:, ok["V1_gb"]] / th[:, ok[f"V1_{rr}"]]; d1 = (1 - R2[:, :, ok["V1_gb"]] / R2[:, :, ok[f"V1_{rr}"]]).mean(1)
    cetak(f"V3 gb atas {rr}", p3, d3)
    cetak(f"V1 gb atas {rr}", p1, d1)
    RINGKAS[f"D_{rr}"] = cetak(f"beda V3-V1 atas {rr}", p3 - p1, d3 - d1)
# Brier GB langsung (tanpa rujukan): BSS V3 atas V1
pt = 1 - th[:, ok["V3_gb"]] / th[:, ok["V1_gb"]]; ds = (1 - R2[:, :, ok["V3_gb"]] / R2[:, :, ok["V1_gb"]]).mean(1)
RINGKAS["D_gb"] = cetak("gb V3 atas gb V1 (Brier)", pt, ds)

print("\nRingkasan: " + "; ".join(
    f"{k} {'ya' if v[0] else 'tidak (' + ', '.join(v[1]) + ')'} [{v[2]}/6 benih; EF-L {v[3]}]"
    for k, v in RINGKAS.items() if not k.startswith("D")) + "; " +
    "; ".join(f"{k} lolos {v}" for k, v in RINGKAS.items() if k.startswith("D")))
print("selesai.")
