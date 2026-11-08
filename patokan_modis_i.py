"""
Revisi fase I (Hasil_Patokan_14): jalur MODIS 2003-2011 dengan enam benih GB, kontrol panjang latih,
latih tanpa 2020-2025, pemecahan geser-tingkat periode/tahun, dan selisih terkecil yang terdeteksi.

Varian lipatan (panel, ambang, fitur, parameter sama dengan patokan_modis_h.py):
  V0 utama            latih = semua tahun MODIS lain, ambang dari 2003-2025
  V1 latih < 2016     latih = 2003-2015 tanpa tahun uji (12 tahun)
  V2 ambang latih     latih seperti V0, ambang dan kabupaten yang dinilai dari tahun latih lipatan
  V3 12 tahun acak    latih = 12 dari 22 tahun lain, lima undian per lipatan (default_rng(2026 + u)), ambang V0
  V4 tanpa 2020-2025  latih = tahun lain 2003-2019 (16 tahun), ambang V0
GB enam benih {42, 1, 2, 3, 4, 5}; Brier GB per tahun = rata-rata atas benih (dan undian V3).

Pakai (satu sasaran per jalan):
    python -u patokan_modis_i.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_i_kab.log
    python -u patokan_modis_i.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_i_nas.log
"""
import csv
import glob
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
KAPPA = 2.0
N = 1000
UJI = list(range(2003, 2012))
CALON = ["konstan", "konstan_bulan", "ef1", "ef3", "ef1_pasca", "ef1_halus"]
BENIH = [42, 1, 2, 3, 4, 5]
UNDIAN = 5
DM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_MODIS")
REF = "ef1" if SAS == "y_kabupaten" else "ef3"
RUJ = ["kalender", "efp", "efl", "lokal"]

# ------------------------------------------------- panel (sama dengan patokan_modis_h.py)
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
      f"benih {BENIH}; undian V3 {UNDIAN}")

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


def X_lokal(g, t, m, ef1, YL1):
    return np.column_stack([logit(ef1), np.nan_to_num(YL1[g, t, m])]
                           + [np.log1p(np.nan_to_num(LAG[k][g, t, m])) for k in (1, 2, 3)]
                           + [SIN[g, t, m], COS[g, t, m]])


def X_gb(g, t, m, ef1, a, YL1):
    return np.column_stack([np.log1p(np.nan_to_num(LAG[k][g, t, m])) for k in (1, 2, 3, 12)]
                           + [SIN[g, t, m], COS[g, t, m], ef1, np.nan_to_num(YL1[g, t, m]), a])


def tahun_latih(u, varian, d=0):
    lain = [x for x in THN if x != u]
    if varian == "V1":
        return [x for x in lain if x < 2016]
    if varian == "V3":
        return sorted(int(x) for x in UNDIAN_TL[u][d])
    if varian == "V4":
        return [x for x in lain if x <= 2019]
    return lain


# undian V3 ditetapkan sekali per lipatan: lima undian berurutan dari satu generator per tahun uji
UNDIAN_TL = {}
for u in UJI:
    rg = np.random.default_rng(2026 + u)
    lain = np.array([x for x in THN if x != u])
    UNDIAN_TL[u] = [rg.choice(lain, 12, replace=False) for _ in range(UNDIAN)]


def lipatan(u, varian, d=0):
    tl = tahun_latih(u, varian, d)
    L, ada = label(THN if varian != "V2" else tl)
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
    lok = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_lokal(g, t, m, rl["ef1"], YL1), yl)
    Xl = X_gb(g, t, m, rl["ef1"], an, YL1)
    g2, t2, m2 = indeks(Y, [u])
    ru = rujukan(Y, tl, g2, t2, m2); au = anomali(tl, t2, m2)
    Xu = X_gb(g2, t2, m2, ru["ef1"], au, YL1)
    b = pd.DataFrame({"gid": GID[g2], "tag": str(u), "y": Y[g2, t2, m2], "kalender": ru[REF], "efp": ru[pilih],
                      "efl": lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), au]))[:, 1],
                      "lokal": lok.predict_proba(X_lokal(g2, t2, m2, ru["ef1"], YL1))[:, 1]})
    for s in BENIH:
        xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                           reg_lambda=1.0, eval_metric="logloss", random_state=s, verbosity=0, n_jobs=-1)
        xg.fit(Xl, yl)
        b[f"gb_{s}"] = xg.predict_proba(Xu)[:, 1]
    print(f"  {varian} {u}" + (f" undian {d}" if varian == "V3" else "") +
          f": latih {len(tl)} tahun ({','.join(str(x)[2:] for x in tl)}), {len(yl):,} baris; uji {len(b):,}, "
          f"prevalensi {b.y.mean() * 100:.2f}%; EF-P {pilih}", flush=True)
    return b


def geser(f, ybar):
    lf = logit(f, 1e-6)
    h = lambda dl: np.mean(1 / (1 + np.exp(-(lf + dl)))) - ybar
    return 1 / (1 + np.exp(-(lf + brentq(h, -30, 30))))


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


# ------------------------------------------------- deskriptif homogenitas (sekali, di jalan per kabupaten)
if SAS == "y_kabupaten":
    hit = Counter()
    for f in sorted(glob.glob(os.path.join(DM, "kelas", "modis_sp_*.csv"))):
        with open(f, encoding="utf-8", newline="") as a:
            for x in csv.DictReader(a):
                if x["type"] == "0" and x["confidence"] != "l":
                    hit[(x["acq_date"][:4], x["satellite"], x["version"])] += 1
    print("\nI3 deskriptif: deteksi MODIS bersaring per tahun (seluruh unduhan, sebelum gabung poligon)")
    for th in sorted({k[0] for k in hit}):
        te = sum(v for k, v in hit.items() if k[0] == th and k[1] == "Terra")
        aq = sum(v for k, v in hit.items() if k[0] == th and k[1] == "Aqua")
        vs = sorted({k[2] for k in hit if k[0] == th})
        print(f"    {th}: Terra {te:>7,}  Aqua {aq:>7,}  bagian Terra {te / max(te + aq, 1):.3f}  versi {','.join(vs)}")

B.cek_ap()
tags = [str(u) for u in UJI]
RINGKAS = {}
for varian in ("V0", "V1", "V2", "V3", "V4"):
    t0 = time.time()
    print("\n" + "=" * 100 + f"\nI {varian}  {SAS}  (rujukan pasangan {REF}), n = 9, df 8\n" + "=" * 100, flush=True)
    nd = UNDIAN if varian == "V3" else 1
    bag = []
    for u in UJI:
        fr = [lipatan(u, varian, d) for d in range(nd)]
        for k in range(1, nd):
            assert np.array_equal(fr[k].y.to_numpy(), fr[0].y.to_numpy()) and np.array_equal(fr[k].gid, fr[0].gid)
        b = fr[0][["gid", "tag", "y"]].copy()
        for d, x in enumerate(fr):
            for c in x.columns[3:]:
                b[f"{c}_d{d}"] = x[c].to_numpy()
        bag.append(b)
    P = pd.concat(bag, ignore_index=True)
    GBC = [f"gb_{s}_d{d}" for s in BENIH for d in range(nd)]
    KOL = [c for c in P.columns if c not in ("gid", "tag", "y")]
    # pemecahan periode/tahun (benih 42, undian 0) untuk V0 dan V1
    PEC = varian in ("V0", "V1")
    if PEC:
        for c in ("kalender_d0", "efp_d0", "gb_42_d0"):
            P[c + "_pd"] = geser(P[c].to_numpy(), P.y.mean())
            P[c + "_lv"] = np.nan
            for _, x in P.groupby("tag"):
                P.loc[x.index, c + "_lv"] = geser(x[c].to_numpy(), x.y.mean())
            KOL += [c + "_pd", c + "_lv"]
    ix = {c: i for i, c in enumerate(KOL)}
    # keluaran fn: Brier rata-rata (rujukan atas undian; gb atas undian x benih; gb per benih), AP (nd = 1)
    KEL = RUJ + ["gb"] + [f"gb{s}" for s in BENIH] + (
        [f"{c}_{w}" for c in ("kalender", "efp", "gb42") for w in ("pd", "lv")] if PEC else [])
    AP = ["kalender", "efl", "gb"] + [f"gb{s}" for s in BENIH] if nd == 1 else []
    ok = {c: i for i, c in enumerate(KEL)}; nk = len(KEL)

    def fn(bk, j):
        y = bk.y[j]
        bs = {c: np.mean((bk.s[c][j] - y) ** 2) for c in KOL}
        out = [np.mean([bs[f"{c}_d{d}"] for d in range(nd)]) for c in RUJ]
        out.append(np.mean([bs[c] for c in GBC]))
        out += [np.mean([bs[f"gb_{s}_d{d}"] for d in range(nd)]) for s in BENIH]
        if PEC:
            out += [bs[f"{c}_d0_{w}"] for c in ("kalender", "efp", "gb_42") for w in ("pd", "lv")]
        ap = {}
        if nd == 1:
            ap = {c: B.ap_cepat(y, bk.s[f"{c}_d0"][j]) for c in ("kalender", "efl")}
            gs = [B.ap_cepat(y, bk.s[f"gb_{s}_d0"][j]) for s in BENIH]
            out += [ap["kalender"], ap["efl"], np.mean(gs)] + gs
        return np.array(out)

    kab = P.gid.unique()
    blok = {t: B.Blok(x, KOL, kab) for t, x in P.groupby("tag")}
    th = np.array([fn(blok[t], slice(None)) for t in tags])
    R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)
    print(f"  ({time.time() - t0:.0f} s)  Brier rata-rata: " + "  ".join(f"{c} {th[:, ok[c]].mean():.5f}" for c in RUJ + ["gb"]))
    bss = lambda a, rr: (1 - th[:, ok[a]] / th[:, ok[rr]], (1 - R2[:, :, ok[a]] / R2[:, :, ok[rr]]).mean(1))
    print("  BSS (gb = rata-rata enam benih" + (", lima undian" if nd > 1 else "") + ")")
    lolos = {}
    for a, rr in (("efl", "kalender"), ("efp", "kalender"), ("efl", "efp"), ("lokal", "efl"),
                  ("gb", "kalender"), ("gb", "efp"), ("gb", "efl")):
        lolos[(a, rr)] = cetak(f"{a} atas {rr}", *bss(a, rr))
    if nd == 1:
        print("  AUC-PR (gb = rata-rata AUC-PR enam benih)")
        for a, rr in (("gb", "efl"), ("gb", "kalender")):
            i, k = nk + AP.index(a), nk + AP.index(rr)
            cetak(f"{a} - {rr}", th[:, i] - th[:, k], (R2[:, :, i] - R2[:, :, k]).mean(1))
    semua = all(lolos[("gb", x)] for x in ("kalender", "efp", "efl"))
    gagal = [x for x in ("kalender", "efp", "efl") if not lolos[("gb", x)]]
    print(f"  R36c {varian} rata-rata benih: kalender {lolos[('gb', 'kalender')]}, EF-P {lolos[('gb', 'efp')]}, "
          f"EF-L {lolos[('gb', 'efl')]} -> mengalahkan ketiga rujukan: {semua}" + (f" (tidak: {', '.join(gagal)})" if gagal else ""))
    print("  Per benih: BSS titik atas kalender / EF-P / EF-L, batas bawah t, R36c")
    nlolos = 0
    for s in BENIH:
        baris, ls = [], []
        for rr in ("kalender", "efp", "efl"):
            pt, ds = bss(f"gb{s}", rr)
            lo = np.nanpercentile(ds, 2.5); tl = t95(pt)[0]
            ls.append(lo > 0 and tl > 0); baris.append(f"{pt.mean():+.4f} (t {tl:+.4f})")
        nlolos += all(ls)
        print(f"    benih {s:>2}: " + " / ".join(baris) + f"  R36c {all(ls)}", flush=True)
    for rr in ("kalender", "efp", "efl"):
        v = [bss(f"gb{s}", rr)[0].mean() for s in BENIH]
        print(f"    rentang benih gb atas {rr:<9} {min(v):+.4f} .. {max(v):+.4f}")
    print(f"    benih yang lolos R36c sendiri: {nlolos} dari {len(BENIH)}", flush=True)
    RINGKAS[varian] = (semua, gagal, nlolos)
    if PEC:
        print("  I4 Pemecahan geser-tingkat (benih 42): mentah / offset periode / per tahun; bagian periode, bagian tahun")
        for acu in ("kalender", "efp"):
            raw, rd = bss("gb42", acu)
            pd_, pdd = (1 - th[:, ok["gb42_pd"]] / th[:, ok[f"{acu}_pd"]],
                        (1 - R2[:, :, ok["gb42_pd"]] / R2[:, :, ok[f"{acu}_pd"]]).mean(1))
            lv, lvd = (1 - th[:, ok["gb42_lv"]] / th[:, ok[f"{acu}_lv"]],
                       (1 - R2[:, :, ok["gb42_lv"]] / R2[:, :, ok[f"{acu}_lv"]]).mean(1))
            bp, bt = 1 - pdd / rd, (pdd - lvd) / rd
            print(f"    gb atas {acu:<9} mentah {raw.mean():+.4f}  periode {pd_.mean():+.4f}  tahun {lv.mean():+.4f}  "
                  f"bagian periode {1 - pd_.mean() / raw.mean():.2f} [{np.nanpercentile(bp, 2.5):.2f}, {np.nanpercentile(bp, 97.5):.2f}]  "
                  f"bagian tahun {(pd_.mean() - lv.mean()) / raw.mean():.2f} [{np.nanpercentile(bt, 2.5):.2f}, {np.nanpercentile(bt, 97.5):.2f}]  "
                  f"total {1 - lv.mean() / raw.mean():.2f}", flush=True)
    if varian == "V0":
        pt = 1 - th[:, ok["gb42"]] / th[:, ok["efl"]]
        print(f"  cek benih 42 = fase H: gb atas efl {pt.mean():+.4f}")
    del P, blok, R2

print("\nRingkasan R36c rata-rata benih: " + "; ".join(
    f"{v} {'ya' if s else 'tidak (' + ', '.join(g) + ')'} [{n}/6 benih]" for v, (s, g, n) in RINGKAS.items()))
print("selesai.")
