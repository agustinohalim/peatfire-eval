"""
PeatFireBench revisi fase F (Hasil_Patokan_11): EF-L sebagai urutan dan sebagai aturan siaga, kuota yang
disetel di tahun latih, dan gradient boosting yang diberi anomali titik panas nasional bulan lalu.

F1  AUC-PR dalam-tahun EF-L (ef1_tingkat) dan ef1_halus atas rujukan pasangan; model - EF-L (R29).
F2  per kabupaten saja: aturan siaga (i) p > a, (ii) kuota 10 %, (iii) ambang kuantil 0,9 tahun latih,
    (iv) ambang nilai maksimum tahun latih, (v) ambang terbaik tahun uji (batas), (vi) kuota disetel
    di tahun latih; untuk xgb, ef1_tingkat, klim_frek1 (R30, R31).
F3  xgb_nas = GB baku + kolom anomali_nasional (R32).
F4  BSS GB, RF atas EF-L per tahun ekstrem dengan selang kabupaten.

Fungsi EF-L (larik, jendela, rujukan, anomali, logit, indeks) disalin dari patokan_revisi_peluang.py,
fungsi siaga (kuota, nilai, ukur, ambang_oof) dari patokan_revisi_siaga2.py, tanpa perubahan isi.

Pakai (satu sasaran per jalan, memori):
    python -u patokan_revisi_tingkat.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_revisi_tingkat_kab.log
    python -u patokan_revisi_tingkat.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_revisi_tingkat_nas.log
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_tingkat_gb_<sasaran>.csv (tidak masuk repo)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
REF = B.PASANGAN[SAS]
KUNCI = ["sasaran", "skema", "tag", "gid", "bulan"]
ALFA = [0.05, 0.1, 0.2, 0.3, 0.5]
KAPPA = 2.0
FOLD = [("S1", u) for u in range(2020, 2026)] + [("S2", u) for u in (2015, 2014, 2019)]
N = 1000

# ------------------------------------------------- EF-L (disalin dari patokan_revisi_peluang.py)
df = pd.read_csv(os.path.join(B.D, "panel_nasional_fitur_penuh.csv"),
                 usecols=["gid", "tahun", "bulan", "bulan_ke", "tp_lag1", "y_kabupaten", "y_gabungan", "ada_positif"])
GID = np.sort(df.gid.unique()); THN = np.arange(df.tahun.min(), df.tahun.max() + 1)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
df["_g"] = df.gid.map(gi); df["_t"] = df.tahun.map(ti); df["_m"] = df.bulan_ke - 1
ADA = df.groupby("_g").ada_positif.max().reindex(range(len(GID))).fillna(0).to_numpy() == 1
NAS = df.groupby(["_t", "_m"]).tp_lag1.sum()
LN = np.full((len(THN), 12), np.nan)
LN[NAS.index.get_level_values(0), NAS.index.get_level_values(1)] = np.log1p(NAS.to_numpy())


def larik(ylab):
    Y = np.full((len(GID), len(THN), 12), np.nan)
    Y[df._g, df._t, df._m] = df[ylab].to_numpy(float)
    if ylab == "y_kabupaten":
        Y[~ADA] = np.nan
    return Y


def jendela(a):
    return a + np.roll(a, 1, axis=-1) + np.roll(a, -1, axis=-1)


def rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=False):
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
    bagi = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), 0.0)
    ef3 = bagi(s3, n3)
    return {"ef1_halus": (s1 + KAPPA * ef3) / (n1 + KAPPA)}


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


def efl_lipatan(Y, skema, uji):
    """EF-L untuk baris latih (nilai pas, kalender dan anomali tinggalkan-tahun-sendiri) dan baris uji."""
    tahun_latih = [x for x in THN if (x < uji if skema == "S1" else x != uji)]
    g, t, m = indeks(Y, tahun_latih)
    rl = rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=True)
    X = np.column_stack([logit(rl["ef1_halus"]), anomali(tahun_latih, t, m, keluar_sendiri=True)])
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(X, Y[g, t, m])
    latih = pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[a]}-{b + 1:02d}" for a, b in zip(t, m)],
                          "tahun_dalam": THN[t], "y": Y[g, t, m], "ef1_tingkat": lg.predict_proba(X)[:, 1]})
    g, t, m = indeks(Y, [uji])
    ru = rujukan(Y, tahun_latih, g, t, m)
    p = lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), anomali(tahun_latih, t, m)]))[:, 1]
    uji_df = pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[a]}-{b + 1:02d}" for a, b in zip(t, m)], "ef1_tingkat_ulang": p})
    return latih, uji_df


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def cetak(nama, pt, dist, kabd=None):
    lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
    kab = "" if kabd is None else " kab [{:+.4f}, {:+.4f}]".format(*np.nanpercentile(kabd, [2.5, 97.5]))
    lolos = lo > 0 and tl > 0
    print(f"    {nama:<40}{pt.mean():+.4f}{kab} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
          f"{'  LOLOS' if lolos else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
    return lolos


Y = larik(SAS)
LATIH_EFL = {}
UJI_EFL = []
for sk, u in FOLD:
    lt, uj = efl_lipatan(Y, sk, u)
    LATIH_EFL[(sk, str(u))] = lt
    uj["skema"], uj["tag"] = sk, str(u)
    UJI_EFL.append(uj)
UJI_EFL = pd.concat(UJI_EFL, ignore_index=True)

# ------------------------------------------------- F3: GB dengan anomali nasional
OUTGB = os.path.join(B.D, f"patokan_revisi_tingkat_gb_{SAS}.csv")
if not os.path.exists(OUTGB):
    D = R.df.copy()
    D["_t"] = D.tahun.map(ti); D["_m"] = D.bulan_ke - 1
    FIT = R.FITUR + ["anomali_nasional"]
    baris = []
    for sk, u in FOLD:
        tl = [x for x in THN if (x < u if sk == "S1" else x != u)]
        latih = D[D.tahun.isin(tl)].copy(); uji = D[D.tahun == u].copy()
        latih["anomali_nasional"] = anomali(tl, latih._t.to_numpy(), latih._m.to_numpy(), keluar_sendiri=True)
        uji["anomali_nasional"] = anomali(tl, uji._t.to_numpy(), uji._m.to_numpy())
        xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                           reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        xg.fit(latih[FIT].fillna(0.0).to_numpy(), latih[SAS].to_numpy())
        b = uji[["gid", "bulan"]].copy()
        b["sasaran"], b["skema"], b["tag"] = SAS, sk, str(u)
        b["xgb_nas"] = xg.predict_proba(uji[FIT].fillna(0.0).to_numpy())[:, 1]
        baris.append(b)
        print(f"F3 {sk} {u}: dilatih {len(latih)} baris", flush=True)
    pd.concat(baris, ignore_index=True).to_csv(OUTGB, index=False)

# ------------------------------------------------- gabung
P = B.muat_P(fwi=False)
P = P[P.sasaran == SAS]
for f, kol in ((f"patokan_revisi_peluang_{SAS}.csv", ["ef1_tingkat", "ef1_halus"]),
               ("patokan_revisi_setel.csv", ["xgb_setel"]), ("patokan_revisi_sederhana.csv", ["log_ef1_jeda"]),
               (os.path.basename(OUTGB), ["xgb_nas"])):
    X = pd.read_csv(os.path.join(B.D, f), dtype={"tag": str})
    X = X[X.sasaran == SAS][KUNCI + kol]
    P = P.merge(X, on=KUNCI, how="left", validate="one_to_one")
P = P.merge(UJI_EFL, on=["skema", "tag", "gid", "bulan"], how="left", validate="one_to_one")
d = B.ambil_sasaran(P, SAS)
d = d[((d.skema == "S1") & d.tag.isin(B.BIASA)) | ((d.skema == "S2") & d.tag.isin(B.EKSTREM))]
B.cek_ap()
print(f"cek EF-L dihitung ulang vs tersimpan: selisih maks {np.abs(d.ef1_tingkat - d.ef1_tingkat_ulang).max():.2e}; "
      f"baris kosong {int(d[['ef1_tingkat', 'xgb_setel', 'log_ef1_jeda', 'xgb_nas']].isna().sum().sum())}", flush=True)

# ------------------------------------------------- F1 dan F3/F4
KOL = ["ef1_tingkat", "ef1_halus", REF, "rf", "xgb", "xgb_setel", "log_ef1_jeda", "xgb_nas"]
ix = {c: i for i, c in enumerate(KOL)}
s1 = d[(d.skema == "S1")]
kab = s1.gid.unique()
blok = {t: B.Blok(x, KOL, kab) for t, x in s1.groupby("tag")}


def fn_ap(b, j):
    y = b.y[j]
    return np.array([B.ap_cepat(y, b.s[c][j]) for c in KOL] + [np.mean((b.s[c][j] - y) ** 2) for c in KOL])


nk = len(KOL)
th = np.array([fn_ap(blok[t], slice(None)) for t in B.BIASA])
R2 = B.dua_tahap(blok, fn_ap, N, np.random.default_rng(B.SEED), kab)


def kab_tahun(blok, fn, n, rng, kab):
    """tahun tetap, kabupaten diambil ulang (tarikan sama untuk semua tahun): (n, tahun, k)."""
    out = []
    for _ in range(n):
        pilih = rng.choice(kab, len(kab))
        out.append([fn(blok[t], blok[t].baris(pilih)) for t in blok])
    return np.array(out)


RK = kab_tahun(blok, fn_ap, N, np.random.default_rng(B.SEED + 1), kab)
print("\n" + "=" * 110 + f"\nF1. {SAS}: AUC-PR dalam-tahun, tahun biasa (rujukan pasangan {B.LABEL[REF]})\n" + "=" * 110)
print("  AP rata-rata tahun: " + "  ".join(f"{c} {th[:, ix[c]].mean():.3f}" for c in KOL))
for a, r in (("ef1_tingkat", REF), ("ef1_halus", REF), ("rf", "ef1_tingkat"), ("xgb", "ef1_tingkat"),
             ("xgb_setel", "ef1_tingkat"), ("log_ef1_jeda", "ef1_tingkat"), ("xgb_nas", "ef1_tingkat"),
             ("xgb_nas", "xgb"), ("xgb_nas", REF)):
    i, k = ix[a], ix[r]
    cetak(f"AP {a} - {B.LABEL.get(r, r)}", th[:, i] - th[:, k], (R2[:, :, i] - R2[:, :, k]).mean(1), (RK[:, :, i] - RK[:, :, k]).mean(1))
print("\n  F3/F4. BSS dalam-tahun (1 - Brier / Brier acuan), tahun biasa")
print("  Brier rata-rata tahun: " + "  ".join(f"{c} {th[:, nk + ix[c]].mean():.5f}" for c in KOL))
for a, r in (("xgb", "ef1_tingkat"), ("rf", "ef1_tingkat"), ("xgb_nas", "ef1_tingkat"), ("xgb_nas", "xgb"),
             ("xgb_nas", REF), ("ef1_tingkat", REF)):
    i, k = nk + ix[a], nk + ix[r]
    cetak(f"BSS {a} atas {B.LABEL.get(r, r)}", 1 - th[:, i] / th[:, k], (1 - R2[:, :, i] / R2[:, :, k]).mean(1),
          (1 - RK[:, :, i] / RK[:, :, k]).mean(1))

print("\n  Tahun ekstrem (S2), selang kabupaten 1.000")
s2 = d[d.skema == "S2"]
for t in B.EKSTREM:
    x = s2[s2.tag == t]; kb = x.gid.unique(); b = B.Blok(x, KOL, kb)
    pt = fn_ap(b, slice(None)); D_ = B.kab_tetap(b, fn_ap, N, np.random.default_rng(B.SEED), kb)
    teks = []
    for a, r in (("ef1_tingkat", REF), ("xgb", "ef1_tingkat"), ("rf", "ef1_tingkat"), ("xgb_nas", "ef1_tingkat")):
        i, k = ix[a], ix[r]; lo, hi = np.nanpercentile(D_[:, i] - D_[:, k], [2.5, 97.5])
        teks.append(f"AP {a}-{B.LABEL.get(r, r)} {pt[i] - pt[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for a, r in (("xgb", "ef1_tingkat"), ("rf", "ef1_tingkat"), ("xgb_nas", "ef1_tingkat"), ("ef1_tingkat", REF)):
        i, k = nk + ix[a], nk + ix[r]; lo, hi = np.nanpercentile(1 - D_[:, i] / D_[:, k], [2.5, 97.5])
        teks.append(f"BSS {a}/{B.LABEL.get(r, r)} {1 - pt[i] / pt[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"    {t}: " + ";  ".join(teks), flush=True)

# ------------------------------------------------- F2 siaga (disalin dari patokan_revisi_siaga2.py)
if SAS != "y_kabupaten":
    print("\nF2 hanya untuk sasaran per kabupaten.\nselesai.")
    sys.exit(0)


class BlokB(B.Blok):
    def __init__(self, g, kolom, kab):
        super().__init__(g, kolom, kab)
        self.bulan = g.bulan.to_numpy()


def kuota(b, j, m, frac=0.1):
    jj = np.arange(len(b.y))[j]
    bl = b.bulan[jj]; s = b.s[m][jj]; siaga = np.zeros(len(jj), bool)
    for mo in np.unique(bl):
        w = np.where(bl == mo)[0]; n = max(1, int(round(frac * len(w))))
        siaga[w[np.argsort(-s[w], kind="stable")[:n]]] = True
    return siaga


def nilai(H, F, s, a):
    e_klim, e_semp = min(a, s), a * s
    return (e_klim - (F * a * (1 - s) + H * a * s + (1 - H) * s)) / (e_klim - e_semp)


def ukur(y, siaga):
    tp = (siaga & (y == 1)).sum()
    H = tp / max(y.sum(), 1); F = (siaga & (y == 0)).sum() / max((y == 0).sum(), 1)
    return [H, 1 - tp / max(siaga.sum(), 1), siaga.mean()] + [nilai(H, F, y.mean(), a) for a in ALFA]


def ambang_oof(o, m, a):
    calon = np.unique(np.quantile(o[m].to_numpy(dtype=float), np.linspace(0, 1, 401)))
    tot = np.zeros(len(calon)); n = 0
    for _, g in o.groupby("tahun_dalam"):
        y = g.y.to_numpy(); p = g[m].to_numpy(dtype=float)
        if y.sum() == 0:
            continue
        pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
        H = 1 - np.searchsorted(pos, calon, side="left") / len(pos)
        F = 1 - np.searchsorted(neg, calon, side="left") / len(neg)
        tot += nilai(H, F, y.mean(), a); n += 1
    return calon[int(np.argmax(tot / n))]


# baru di fase F
QGRID = np.round(np.arange(0.02, 0.301, 0.02), 2)


def hf(y, siaga):
    return (siaga & (y == 1)).sum() / max(y.sum(), 1), (siaga & (y == 0)).sum() / max((y == 0).sum(), 1)


def kuota_setel(o, m, a):
    """bagian kuota bulanan dengan nilai rata-rata per tahun tertinggi di tahun latih."""
    tot = np.zeros(len(QGRID)); n = 0
    for _, g in o.groupby("tahun_dalam"):
        y = g.y.to_numpy()
        if y.sum() == 0:
            continue
        p = g[m].to_numpy(dtype=float); bl = g.bulan.to_numpy()
        urut = {mo: np.where(bl == mo)[0][np.argsort(-p[bl == mo], kind="stable")] for mo in np.unique(bl)}
        for qi, q in enumerate(QGRID):
            siaga = np.zeros(len(y), bool)
            for w in urut.values():
                siaga[w[:max(1, int(round(q * len(w))))]] = True
            tot[qi] += nilai(*hf(y, siaga), y.mean(), a)
        n += 1
    return float(QGRID[int(np.argmax(tot / n))])


def terbaik(y, p, a):
    calon = np.unique(p)
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    H = 1 - np.searchsorted(pos, calon, side="left") / len(pos)
    F = 1 - np.searchsorted(neg, calon, side="left") / len(neg)
    return float(np.max(nilai(H, F, y.mean(), a)))


O = pd.read_csv(f"{B.D}/patokan_revisi_oof.csv", dtype={"tag": str})
O = O[(O.ada_positif == 1) & (O.sasaran == "y_kabupaten")]
MK = ["xgb", "ef1_tingkat", "klim_frek1"]
SETEL = ["xgb", "ef1_tingkat"]
KEL = {"biasa": [("S1", t) for t in B.BIASA], "ekstrem": [("S2", t) for t in B.EKSTREM]}
UK = ["hit", "FAR", "siaga"] + [f"REV{a}" for a in ALFA]
nama = ([f"kuota {m} {u}" for m in MK for u in UK] + [f"amb10 {m} {u}" for m in MK for u in UK]
        + [f"{r} {m} REV{a}" for r in ("pa", "ambnilai") for m in MK for a in ALFA]
        + [f"setel {m} REV{a}" for m in SETEL for a in ALFA])
iz = {n_: i for i, n_ in enumerate(nama)}
print("\n" + "=" * 110 + "\nF2. Siaga per kabupaten: EF-L, GB, EF-1m\n" + "=" * 110)
for kel, kunci in KEL.items():
    blk = {k: BlokB(d[(d.skema == k[0]) & (d.tag == k[1])], MK, kab) for k in kunci}
    latih = {}
    for k in kunci:
        o = O[(O.skema == k[0]) & (O.tag == k[1])][["gid", "bulan", "tahun_dalam", "y", "xgb", "klim_frek1"]]
        e = LATIH_EFL[k][["gid", "bulan", "tahun_dalam", "y", "ef1_tingkat"]]
        latih[k] = (o, e)
    sumber = lambda k, m: latih[k][1] if m == "ef1_tingkat" else latih[k][0]
    thr10 = {(k, m): np.quantile(sumber(k, m)[m].to_numpy(dtype=float), 0.9) for k in kunci for m in MK}
    thrv = {(k, m, a): ambang_oof(sumber(k, m), m, a) for k in kunci for m in MK for a in ALFA}
    qset = {(k, m, a): kuota_setel(sumber(k, m), m, a) for k in kunci for m in SETEL for a in ALFA}
    print(f"\n--- {kel}: bagian kuota disetel (per lipatan, a = {ALFA}):")
    for m in SETEL:
        print(f"    {m:<12}" + "  ".join(f"{k[1]} " + "/".join(f"{qset[(k, m, a)]:.2f}" for a in ALFA) for k in kunci))
    for k, b in blk.items():
        b.kunci = k

    def fn(b, j):
        y = b.y[j]; out = []
        for m in MK:
            out += ukur(y, kuota(b, j, m))
        for m in MK:
            out += ukur(y, b.s[m][j] >= thr10[(b.kunci, m)])
        for m in MK:
            out += [nilai(*hf(y, b.s[m][j] > a), y.mean(), a) for a in ALFA]
        for m in MK:
            out += [nilai(*hf(y, b.s[m][j] >= thrv[(b.kunci, m, a)]), y.mean(), a) for a in ALFA]
        for m in SETEL:
            out += [nilai(*hf(y, kuota(b, j, m, qset[(b.kunci, m, a)])), y.mean(), a) for a in ALFA]
        return np.array(out, dtype=float)

    th2 = np.array([fn(blk[k], slice(None)) for k in kunci])
    t2 = th2.mean(0)
    print(f"    prevalensi {np.mean([blk[k].y.mean() for k in kunci]) * 100:.2f}%")
    print(f"    {'aturan':<26}{'hit':>7}{'FAR':>7}{'siaga':>8}   REV a = " + " / ".join(str(a) for a in ALFA))
    for aturan in ("kuota", "amb10"):
        for m in MK:
            print(f"    {aturan + ' ' + m:<26}{t2[iz[f'{aturan} {m} hit']]:>7.3f}{t2[iz[f'{aturan} {m} FAR']]:>7.3f}"
                  f"{t2[iz[f'{aturan} {m} siaga']] * 100:>7.1f}%   " + " / ".join(f"{t2[iz[f'{aturan} {m} REV{a}']]:+.3f}" for a in ALFA))
    for aturan, ms in (("pa", MK), ("ambnilai", MK), ("setel", SETEL)):
        for m in ms:
            print(f"    {aturan + ' ' + m:<26}{'':>22}   " + " / ".join(f"{t2[iz[f'{aturan} {m} REV{a}']]:+.3f}" for a in ALFA))
    for m in MK:
        bt = np.array([[terbaik(blk[k].y, blk[k].s[m], a) for a in ALFA] for k in kunci]).mean(0)
        print(f"    {'terbaik ' + m:<26}{'':>22}   " + " / ".join(f"{v:+.3f}" for v in bt))
    if kel == "biasa":
        R3 = B.dua_tahap(blk, fn, N, np.random.default_rng(B.SEED), kab)
        print("\n  R30. GB - EF-L, nilai ekonomi relatif")
        for aturan in ("pa", "ambnilai"):
            for a in ALFA:
                i, k = iz[f"{aturan} xgb REV{a}"], iz[f"{aturan} ef1_tingkat REV{a}"]
                cetak(f"{aturan} xgb - EF-L a={a}", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
        print("\n  R31. Kuota disetel - ambang nilai maksimum (aturan iv), model sama")
        for m in SETEL:
            for a in ALFA:
                i, k = iz[f"setel {m} REV{a}"], iz[f"ambnilai {m} REV{a}"]
                cetak(f"setel {m} - ambnilai a={a}", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
        print("\n  Tambahan: kuota 10 % hit GB - EF-L, kuota 10 % - kuota disetel GB")
        i, k = iz["kuota xgb hit"], iz["kuota ef1_tingkat hit"]
        cetak("hit kuota xgb - EF-L", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
        for a in ALFA:
            i, k = iz[f"kuota xgb REV{a}"], iz[f"setel xgb REV{a}"]
            cetak(f"kuota10 xgb - setel a={a}", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
    else:
        for aturan in ("pa", "ambnilai", "setel"):
            print(f"    {aturan} xgb - EF-L per tahun: " + "  ".join(
                f"{k[1]} " + "/".join(f"{x[iz[f'{aturan} xgb REV{a}']] - x[iz[f'{aturan} ef1_tingkat REV{a}']]:+.3f}" for a in ALFA)
                for k, x in zip(kunci, th2)))
print("\nselesai.")
