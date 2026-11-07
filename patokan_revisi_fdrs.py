"""
PeatFireBench revisi fase D4 dan D5 (Hasil_Patokan_9).

D4 baseline gaya FDRS dari DC dan FFMC bulanan GFWED MERRA2.CORRECTED, bulan sebelumnya:
  dc_nasional, dc_persentil (dalam sebaran tahun latih kabupaten sendiri), ffmc_nasional, dan
  log_ef1_fdrs (logit EF-1m + DC + FFMC + musim).
D5 pemisahan kebakaran bulan lalu dari cuaca bulan lalu, semuanya logit EF-1m + musim, ditambah:
  ef1_api (y bulan lalu, log1p tp_lag1), ef1_cuaca (DC, FFMC, hujan_lag1, hujan_anom_lag1),
  ef1_keduanya; dan xgb_tanpa_api (GB baku tanpa tp_lag1..3, tp_lag12).
EF-1m baris latih dihitung tanpa tahun baris itu sendiri (seperti log_ef1_jeda, fase C6).

Pakai:    python -u patokan_revisi_fdrs.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_fdrs.log
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_fdrs.csv (tidak masuk repo)
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R
from kelompok_fitur import KELOMPOK

warnings.filterwarnings("ignore")
OUT = f"{B.D}/patokan_revisi_fdrs.csv"
df = R.df.sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
C = pd.read_csv(f"{B.D}/gfwed_komponen_MERRA2.CORRECTED.csv", usecols=["GID_2", "bulan", "dc", "ffmc"])
C = C.rename(columns={"GID_2": "gid"}).sort_values(["gid", "bulan"])
C["dc_lag1"] = C.groupby("gid").dc.shift(1); C["ffmc_lag1"] = C.groupby("gid").ffmc.shift(1)
df = df.merge(C[["gid", "bulan", "dc_lag1", "ffmc_lag1"]], on=["gid", "bulan"], how="left", validate="one_to_one")
print(f"baris panel {len(df)}; DC bulan lalu hilang {df.dc_lag1.isna().mean() * 100:.2f}% (diisi median kabupaten)", flush=True)
for c in ("dc_lag1", "ffmc_lag1"):
    df[c] = df[c].fillna(df.groupby("gid")[c].transform("median")).fillna(df[c].median())
for ylab in B.PASANGAN:
    df[ylab + "_lag1"] = df.groupby("gid")[ylab].shift(1).fillna(0).to_numpy()
TANPA_API = [f for f in R.FITUR if f not in KELOMPOK["api"]]


def ef1_tanpa_tahun_sendiri(latih, ylab):
    tab = R.tabel_klim(latih, ylab)
    s, n = R.ambil(tab, latih, 0)
    x = latih[ylab].to_numpy()
    f = np.where(n > 1, (s - x) / np.maximum(n - 1, 1), 0.0)
    k = R.klim_tanpa_tahun_sendiri(latih)
    return f + 1e-6 * k / (1 + k)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def kol(d, nama, ylab):
    if nama == "api":
        return [d[ylab + "_lag1"].to_numpy(), np.log1p(d.tp_lag1.fillna(0.0).to_numpy())]
    if nama == "cuaca":
        return [d.dc_lag1.to_numpy(), d.ffmc_lag1.to_numpy(), d.hujan_lag1.fillna(0.0).to_numpy(),
                d.hujan_anom_lag1.fillna(0.0).to_numpy()]
    if nama == "fdrs":
        return [d.dc_lag1.to_numpy(), d.ffmc_lag1.to_numpy()]


def X(d, ef1, ylab, kelompok):
    c = [logit(ef1), d.bulan_sin.to_numpy(), d.bulan_cos.to_numpy()]
    for k in kelompok:
        c += kol(d, k, ylab)
    return np.column_stack(c)


SPEK = {"log_ef1_fdrs": ["fdrs"], "ef1_api": ["api"], "ef1_cuaca": ["cuaca"], "ef1_keduanya": ["api", "cuaca"]}


def persentil(latih, uji, c):
    out = np.zeros(len(uji))
    ref = {g: np.sort(x[c].to_numpy()) for g, x in latih.groupby("gid")}
    for g, idx in uji.groupby("gid").groups.items():
        r = ref.get(g)
        v = uji.loc[idx, c].to_numpy()
        out[uji.index.get_indexer(idx)] = np.searchsorted(r, v, side="right") / len(r) if r is not None else 0.5
    return out


def baru(latih, uji, ylab):
    out = {"dc_nasional": uji.dc_lag1.to_numpy(), "ffmc_nasional": uji.ffmc_lag1.to_numpy(),
           "dc_persentil": persentil(latih, uji, "dc_lag1")}
    ef_l = ef1_tanpa_tahun_sendiri(latih, ylab); ef_u = R.klim_frek(latih, uji, ylab, 0)
    for nama, kel in SPEK.items():
        lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        lg.fit(X(latih, ef_l, ylab, kel), latih[ylab].to_numpy())
        out[nama] = lg.predict_proba(X(uji, ef_u, ylab, kel))[:, 1]
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(latih[TANPA_API].fillna(0.0).to_numpy(), latih[ylab].to_numpy())
    out["xgb_tanpa_api"] = xg.predict_proba(uji[TANPA_API].fillna(0.0).to_numpy())[:, 1]
    return out


if not os.path.exists(OUT):
    baris = []
    for sk, tahun in (("S1", range(2020, 2026)), ("S2", (2015, 2014, 2019))):
        for th in tahun:
            latih, uji = (df[df.tahun < th] if sk == "S1" else df[df.tahun != th]), df[df.tahun == th].reset_index(drop=True)
            for ylab in B.PASANGAN:
                b = uji[["gid", "bulan"]].copy()
                b["sasaran"], b["skema"], b["tag"] = ylab, sk, str(th)
                for k, v in baru(latih, uji, ylab).items():
                    b[k] = v
                baris.append(b)
            print(f"{sk} {th} selesai", flush=True)
    pd.concat(baris, ignore_index=True).to_csv(OUT, index=False)
N = pd.read_csv(OUT, dtype={"tag": str})
BARU = [c for c in N.columns if c not in ("gid", "bulan", "sasaran", "skema", "tag")]
B.cek_ap()
P = B.muat_P(fwi=True)
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
S = pd.read_csv(f"{B.D}/patokan_revisi_sederhana.csv", dtype={"tag": str})
P = P.merge(S[kunci + ["log_ef1_jeda"]], on=kunci, how="left", validate="one_to_one")
P = P.merge(N, on=kunci, how="inner", validate="one_to_one")
print(f"baris gabung {len(P)}")


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


print("=" * 100 + "\nD4/D5. Baseline FDRS dan pemisahan jeda\n" + "=" * 100)
for ylab, ref in B.PASANGAN.items():
    d = B.ambil_sasaran(P, ylab)
    s1 = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
    M = BARU + ["fwi", "fwi_persentil", "klim_fwi", "log_ef1_jeda", "xgb", "rf", ref]
    assert s1[M].notna().all().all(), s1[M].isna().sum()
    kab = s1.gid.unique()
    blok = {t: B.Blok(g, M, kab) for t, g in s1.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.array([B.ap_cepat(y, b.s[m][j]) for m in M])

    th = np.array([fn(blok[t], slice(None)) for t in B.BIASA])
    Rr = B.dua_tahap(blok, fn, 1000, np.random.default_rng(B.SEED), kab)
    ix = {m: i for i, m in enumerate(M)}
    print(f"\n--- {ylab}, rujukan {B.LABEL[ref]} (AP {th[:, ix[ref]].mean():.3f}), {len(kab)} kabupaten")

    def beda(a, r):
        pt = th[:, ix[a]] - th[:, ix[r]]; dist = (Rr[:, :, ix[a]] - Rr[:, :, ix[r]]).mean(1)
        lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, thh = t95(pt)
        lolos = lo > 0 and tl > 0
        print(f"  {a + ' - ' + (B.LABEL.get(r, r)):<34}{pt.mean():+.3f}  2tahap [{lo:+.3f}, {hi:+.3f}]  t95 [{tl:+.3f}, {thh:+.3f}]"
              f"  {'LOLOS R16' if lolos else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
        return pt.mean()

    for m in BARU + ["fwi", "fwi_persentil", "klim_fwi", "log_ef1_jeda"]:
        print(f"  AP {m:<20}{th[:, ix[m]].mean():.3f}")
    g = {m: beda(m, ref) for m in BARU + ["fwi", "fwi_persentil", "klim_fwi", "log_ef1_jeda", "xgb"]}
    print("  -- DC/FFMC lawan FWI komposit")
    for a, r in (("dc_nasional", "fwi"), ("dc_persentil", "fwi_persentil"), ("ffmc_nasional", "fwi"),
                 ("log_ef1_fdrs", "klim_fwi"), ("log_ef1_fdrs", "log_ef1_jeda")):
        beda(a, r)
    print("  -- pemisahan jeda")
    for a, r in (("ef1_keduanya", "ef1_api"), ("ef1_keduanya", "ef1_cuaca"), ("xgb", "xgb_tanpa_api"), ("xgb", "ef1_keduanya")):
        beda(a, r)
    tot = g["ef1_keduanya"]
    print(f"  R19: ef1_api / ef1_keduanya = {g['ef1_api'] / tot if tot else np.nan:.2f}; "
          f"ef1_cuaca / ef1_keduanya = {g['ef1_cuaca'] / tot if tot else np.nan:.2f}")
    s2 = d[d.skema == "S2"]
    for t in B.EKSTREM:
        x = s2[s2.tag == t]; r0 = B.ap_cepat(x.y, x[ref])
        print(f"  S2 {t} (AP {B.LABEL[ref]} {r0:.3f}): " + "  ".join(
            f"{m} {B.ap_cepat(x.y, x[m]) - r0:+.3f}" for m in BARU + ["fwi", "log_ef1_jeda", "xgb"]))
print("\nselesai.")
