"""
PeatFireBench revisi fase B2 — kalibrasi ulang dari tahun lampau (Hasil_Patokan_7, R7).

Gradient boosting, per kabupaten. Pemeta dilatih pada prakiraan luar-lipatan tahun latih
(patokan_revisi_oof.csv) dan diterapkan pada tahun uji: Platt, isotonik, Platt + ONI/DMI jeda 1.
ECE 10 bin, Brier, nilai "siaga bila p > a" dan selisih ke amplop, rata-rata tahun per kelompok.

Pakai:    python patokan_revisi_kalibrasi.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_kalibrasi.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
ALFA = [0.05, 0.1, 0.2, 0.3, 0.5]
KEL = {"biasa (S1 2020-2025)": [("S1", str(t)) for t in range(2020, 2026)],
       "ekstrem (S2 2014, 2015, 2019)": [("S2", "2014"), ("S2", "2015"), ("S2", "2019")]}
iklim = pd.read_csv(os.path.join(D, "panel_nasional_fitur_penuh.csv"), usecols=["gid", "bulan", "oni_lag1", "dmi_lag1"])
P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
P = P[(P.sasaran == "y_kabupaten") & (P.ada_positif == 1)].merge(iklim, on=["gid", "bulan"], how="left")
O = pd.read_csv(os.path.join(D, "patokan_revisi_oof.csv"), dtype={"tag": str})
O = O[O.ada_positif == 1].merge(iklim, on=["gid", "bulan"], how="left")
assert P[["oni_lag1", "dmi_lag1"]].notna().all().all() and O[["oni_lag1", "dmi_lag1"]].notna().all().all()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())


def nilai(H, F, s, a):
    e_klim, e_semp = min(a, s), a * s
    return (e_klim - (F * a * (1 - s) + H * a * s + (1 - H) * s)) / (e_klim - e_semp)


def pada_peluang(y, p, a):
    siaga = p > a
    return nilai((siaga & (y == 1)).sum() / max(y.sum(), 1), (siaga & (y == 0)).sum() / max((y == 0).sum(), 1), y.mean(), a)


def amplop(y, p, a):
    o = np.argsort(-p, kind="stable"); ys, ps = y[o], p[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    u = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]
    H, F = np.r_[0, tp[u] / max(ys.sum(), 1)], np.r_[0, fp[u] / max((1 - ys).sum(), 1)]
    return nilai(H, F, y.mean(), a).max()


def pemeta(o):
    y, p = o.y.to_numpy(), o.xgb.to_numpy()
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
    X = np.c_[logit(p), o.oni_lag1, o.dmi_lag1]
    pi = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    return {"mentah": lambda d: d.xgb.to_numpy(),
            "platt": lambda d: platt.predict_proba(logit(d.xgb.to_numpy())[:, None])[:, 1],
            "isotonik": lambda d: iso.predict(d.xgb.to_numpy()),
            "platt_iklim": lambda d: pi.predict_proba(np.c_[logit(d.xgb.to_numpy()), d.oni_lag1, d.dmi_lag1])[:, 1]}


VARIAN = ["mentah", "platt", "isotonik", "platt_iklim"]
for kel, kunci in KEL.items():
    hasil = {v: [] for v in VARIAN}
    for sk, t in kunci:
        u = P[(P.skema == sk) & (P.tag == t)]
        f = pemeta(O[(O.skema == sk) & (O.tag == t)])
        y = u.y.to_numpy()
        for v in VARIAN:
            p = f[v](u)
            hasil[v].append([ece(y, p), np.mean((p - y) ** 2), p.mean(), y.mean()]
                            + [pada_peluang(y, p, a) for a in ALFA] + [amplop(y, p, a) for a in ALFA])
    print(f"\n=== {kel}")
    print(f"  {'varian':<12}{'ECE':>7}{'Brier':>8}{'p rata':>8}{'prev':>7}   nilai p>a (a={ALFA})          selisih ke amplop")
    for v in VARIAN:
        r = np.array(hasil[v]); m = r.mean(0)
        nv, na = m[4:9], m[9:14]
        print(f"  {v:<12}{m[0]:>7.3f}{m[1]:>8.4f}{m[2]:>8.3f}{m[3]:>7.3f}   " + " ".join(f"{x:+.3f}" for x in nv)
              + "   " + " ".join(f"{x:.3f}" for x in na - nv))
    print("  ECE per tahun: " + "; ".join(f"{v} " + " ".join(f"{x:.3f}" for x in np.array(hasil[v])[:, 0]) for v in VARIAN))
print("\nselesai.")
