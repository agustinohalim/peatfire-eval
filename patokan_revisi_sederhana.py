"""
PeatFireBench revisi fase C6 (Hasil_Patokan_8): dua baseline sederhana yang lebih kuat.

  log_ef1_jeda  regresi logistik terskala: logit(EF-1m), y bulan lalu (sasaran sama), log1p tp_lag1..3,
                bulan_sin, bulan_cos. EF-1m baris latih dihitung tanpa tahun baris itu sendiri.
  xgb_sejarah   gradient boosting (parameter patokan_prediksi.py) pada tp_lag1..3, tp_lag12, bulan_sin/cos.

S1 2020-2025 dan S2 2015, 2014, 2019, kedua sasaran, benih 42.

Pakai:    python -u patokan_revisi_sederhana.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_sederhana.log
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_sederhana.csv (tidak masuk repo)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R

warnings.filterwarnings("ignore")
SEJARAH = ["tp_lag1", "tp_lag2", "tp_lag3", "tp_lag12", "bulan_sin", "bulan_cos"]
df = R.df.sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
for ylab in B.PASANGAN:
    df[ylab + "_lag1"] = df.groupby("gid")[ylab].shift(1).fillna(0).to_numpy()
B.cek_ap()


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


def X_ef1(d, ef1, ylab):
    return np.column_stack([logit(ef1), d[ylab + "_lag1"].to_numpy()]
                           + [np.log1p(d[c].fillna(0.0).to_numpy()) for c in ("tp_lag1", "tp_lag2", "tp_lag3")]
                           + [d.bulan_sin.to_numpy(), d.bulan_cos.to_numpy()])


def baru(latih, uji, ylab):
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    lg.fit(X_ef1(latih, ef1_tanpa_tahun_sendiri(latih, ylab), ylab), latih[ylab].to_numpy())
    p_lg = lg.predict_proba(X_ef1(uji, R.klim_frek(latih, uji, ylab, 0), ylab))[:, 1]
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(latih[SEJARAH].fillna(0.0).to_numpy(), latih[ylab].to_numpy())
    return {"log_ef1_jeda": p_lg, "xgb_sejarah": xg.predict_proba(uji[SEJARAH].fillna(0.0).to_numpy())[:, 1]}


baris = []
for sk, tahun in (("S1", range(2020, 2026)), ("S2", (2015, 2014, 2019))):
    for th in tahun:
        latih, uji = (df[df.tahun < th] if sk == "S1" else df[df.tahun != th]), df[df.tahun == th]
        for ylab in B.PASANGAN:
            b = uji[["gid", "bulan"]].copy()
            b["sasaran"], b["skema"], b["tag"] = ylab, sk, str(th)
            for k, v in baru(latih, uji, ylab).items():
                b[k] = v
            baris.append(b)
        print(f"{sk} {th} selesai", flush=True)
N = pd.concat(baris, ignore_index=True)
N.to_csv(f"{B.D}/patokan_revisi_sederhana.csv", index=False)
P = B.muat_P(fwi=False)
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = P.merge(N, on=kunci, how="inner", validate="one_to_one")
BARU = ["log_ef1_jeda", "xgb_sejarah"]

print("=" * 100 + "\nC6. Baseline sederhana yang lebih kuat\n" + "=" * 100)
for ylab, ref in B.PASANGAN.items():
    d = B.ambil_sasaran(P, ylab)
    s1 = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
    kol = BARU + ["xgb", "rf", "log_klim_jeda", ref]
    kab = s1.gid.unique()
    blok = {t: B.Blok(g, kol, kab) for t, g in s1.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.array([B.ap_cepat(y, b.s[m][j]) for m in kol])

    t0 = np.mean([fn(b, slice(None)) for b in blok.values()], axis=0)
    rng = np.random.default_rng(B.SEED)
    kb = B.ulang(blok, fn, 1000, False, rng, kab); dt = B.ulang(blok, fn, 1000, True, rng, kab)
    ir = kol.index(ref)
    print(f"\n--- {ylab}, rujukan {B.LABEL[ref]} (AP {t0[ir]:.3f})")

    def baris_(nama, i, r):
        print(f"  {nama:<34}{B.fmt(t0[i] - t0[r], *B.ci(kb[:, i] - kb[:, r]))}  dua tahap "
              f"[{B.ci(dt[:, i] - dt[:, r])[0]:+.3f}, {B.ci(dt[:, i] - dt[:, r])[1]:+.3f}]", flush=True)

    for m in BARU + ["log_klim_jeda"]:
        i = kol.index(m)
        print(f"  {m:<14} AP {t0[i]:.3f}  per tahun " + " ".join(
            f"{fn(blok[t], slice(None))[i] - fn(blok[t], slice(None))[ir]:+.3f}" for t in B.BIASA))
        baris_(f"{m} - {B.LABEL[ref]}", i, ir)
    for a in ("xgb", "rf"):
        for m in BARU:
            baris_(f"{a} - {m}", kol.index(a), kol.index(m))
    s2 = d[d.skema == "S2"]
    for t in B.EKSTREM:
        g = s2[s2.tag == t]
        print(f"  S2 {t}: " + "  ".join(f"{m} {B.ap_cepat(g.y, g[m]) - B.ap_cepat(g.y, g[ref]):+.3f}" for m in BARU + ["xgb"]))
print("\nselesai.")
