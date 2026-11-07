"""
PeatFireBench revisi fase B3 — prakiraan tanpa data bulan lalu (jeda dua bulan) (Hasil_Patokan_7).

rf dan xgb dilatih ulang (parameter dan benih sama) tanpa fitur jeda satu bulan. S1 2019-2025 dan S2,
kedua sasaran. Dibandingkan dengan rujukan klimatologi terkuat per sasaran dan dengan model jeda satu
bulan (patokan_revisi_prediksi.csv).

Pakai:    python patokan_revisi_jeda2.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_jeda2.log
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_jeda2.csv (tidak masuk repo)
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score as ap
from xgboost import XGBClassifier

import patokan_revisi_prediksi as R

BUANG = ["tp_lag1", "oni_lag1", "dmi_lag1", "hujan_lag1", "hujan_anom_lag1", "fwi_lag1", "fwi_maks_lag1", "fwi_anom_lag1"]
FITUR2 = [f for f in R.FITUR if f not in BUANG]
assert len(FITUR2) == len(R.FITUR) - len(BUANG), set(BUANG) - set(R.FITUR)
RUJUKAN = {"y_kabupaten": "klim_frek1", "y_gabungan": "klim_frek"}
BIASA = [str(t) for t in range(2020, 2026)]
SEED = 2026
df = R.df
print(f"fitur jeda >= 2: {len(FITUR2)} dari {len(R.FITUR)}")


def latih(lt, uj, ylab):
    Xl, Xu, yl = lt[FITUR2].fillna(0.0).to_numpy(), uj[FITUR2].fillna(0.0).to_numpy(), lt[ylab].to_numpy()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(Xl, yl)
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1).fit(Xl, yl)
    return {"rf_j2": rf.predict_proba(Xu)[:, 1], "xgb_j2": xg.predict_proba(Xu)[:, 1]}


baris = []
for ylab in RUJUKAN:
    lip = [("S1", str(t), df[df.tahun < t], df[df.tahun == t]) for t in range(2019, 2026)]
    lip += [("S2", str(t), df[df.tahun != t], df[df.tahun == t]) for t in (2015, 2014, 2019)]
    for sk, tag, lt, uj in lip:
        b = uj[["gid", "bulan"]].copy(); b["sasaran"], b["skema"], b["tag"] = ylab, sk, tag
        for k, v in latih(lt, uj, ylab).items():
            b[k] = v
        baris.append(b)
        print(f"  {ylab} {sk} {tag}", flush=True)
J = pd.concat(baris, ignore_index=True)
J.to_csv(os.path.join(R.D, "patokan_revisi_jeda2.csv"), index=False)
P = pd.read_csv(os.path.join(R.D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
P = P.merge(J, on=["sasaran", "skema", "tag", "gid", "bulan"], how="inner", validate="one_to_one")


def dalam(d, a, b=None):
    return float(np.mean([ap(g.y, g[a]) - (ap(g.y, g[b]) if b else 0) for _, g in d.groupby("tag") if g.y.sum()]))


def boot(d, a, b, n):
    uk = d.gid.unique(); rng = np.random.default_rng(SEED)
    per = {t: g for t, g in d.groupby("tag")}
    idx = {t: {k: np.where(g.gid.to_numpy() == k)[0] for k in uk} for t, g in per.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, len(uk)); v = []
        for t, g in per.items():
            j = np.concatenate([idx[t][k] for k in pilih]); y = g.y.to_numpy()[j]
            if y.sum():
                v.append(ap(y, g[a].to_numpy()[j]) - ap(y, g[b].to_numpy()[j]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


def fmt(x, lo, hi):
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


for sas, ref in RUJUKAN.items():
    d0 = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    b = d0[(d0.skema == "S1") & d0.tag.isin(BIASA)]
    print(f"\n=== {sas}, rujukan {ref} (AUC-PR dalam-tahun {dalam(b, ref):.3f}); S1 2020-2025")
    for m in ("rf", "xgb"):
        m2 = f"{m}_j2"
        print(f"  {m2:<7} AUC-PR {dalam(b, m2):.3f} (jeda 1: {dalam(b, m):.3f})  atas rujukan {fmt(dalam(b, m2, ref), *boot(b, m2, ref, 1000))}"
              f"  rugi lawan jeda 1 {fmt(dalam(b, m, m2), *boot(b, m, m2, 1000))}")
    print("  S2, xgb_j2 atas rujukan [95%, 500]:")
    for t in ("2015", "2014", "2019"):
        dd = d0[(d0.skema == "S2") & (d0.tag == t)]
        print(f"    {t}: {fmt(dalam(dd, 'xgb_j2', ref), *boot(dd, 'xgb_j2', ref, 500))}  (jeda 1: {dalam(dd, 'xgb', ref):+.3f})")
print("\nselesai.")
