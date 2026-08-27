"""
Percobaan 3: menambahkan Indian Ocean Dipole, dan sensitivitas ambang keparahan.

Dua pertanyaan yang belum dijawab naskah:
  A. Apakah DMI memperbaiki penjelasan, mengingat 2014 dan 2019 besar dengan ENSO lemah?
  B. Apakah temuan Percobaan 1 dan 2 bertahan pada ambang persentil 80 dan 95?

Pakai:
    python percobaan3_dmi_ambang.py DL_FIRE_SV-C2_792597/panel_bulanan.csv \\
        DL_FIRE_SV-C2_792597/oni.ascii.txt DL_FIRE_SV-C2_792597/dmi.had.long.data
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py")).read().split("if __name__")[0])

panel_path, oni_path, dmi_path = sys.argv[1], sys.argv[2], sys.argv[3]


# --------------------------------------------------------------- muat DMI

def muat_dmi(path):
    """Berkas NOAA PSL: baris pertama rentang tahun, lalu tahun + 12 nilai bulanan."""
    nilai = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 13:
                continue
            try:
                th = int(p[0])
            except ValueError:
                continue
            if not (1870 <= th <= 2030):
                continue
            for m, v in enumerate(p[1:], 1):
                x = float(v)
                if x < -90:
                    continue
                nilai[f"{th}-{m:02d}"] = x
    return nilai


dmi = muat_dmi(dmi_path)
print(f"DMI dimuat: {len(dmi)} bulan, {min(dmi)} s/d {max(dmi)}")


def musim(nilai, th, bulan_pusat):
    """Rata-rata tiga bulan berpusat di bulan_pusat, meniru definisi musim ONI."""
    v = []
    for off in (-1, 0, 1):
        m = bulan_pusat + off
        y = th
        if m < 1:
            m += 12; y -= 1
        if m > 12:
            m -= 12; y += 1
        k = f"{y}-{m:02d}"
        if k in nilai:
            v.append(nilai[k])
    return float(np.mean(v)) if v else np.nan


# --------------------------------------------------------------- A. tahunan

df, ambang90 = muat(panel_path, oni_path)
tahunan = df.groupby("tahun", observed=True)["titik_panas"].sum()

oni_map = {}
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
with open(oni_path) as f:
    next(f)
    for line in f:
        p = line.split()
        if len(p) >= 4 and p[0] in SEASONS:
            oni_map[(int(p[1]), SEASONS.index(p[0]) + 1)] = float(p[3])

rows = []
for th in tahunan.index:
    rows.append({
        "tahun": th,
        "tp": tahunan[th],
        "oni_aso": oni_map.get((th, 9), np.nan),
        "oni_mjj": oni_map.get((th, 6), np.nan),
        "dmi_aso": musim(dmi, th, 9),
        "dmi_mjj": musim(dmi, th, 6),
    })
T = pd.DataFrame(rows).dropna()

print("\n" + "=" * 74)
print("A. TAHUNAN — apakah DMI menambah penjelasan?")
print("=" * 74)
print(f"{'tahun':<7}{'titik panas':>12}{'ONI ASO':>10}{'DMI ASO':>10}{'ONI MJJ':>10}{'DMI MJJ':>10}")
print("-" * 74)
for _, r in T.sort_values("tp", ascending=False).iterrows():
    print(f"{int(r['tahun']):<7}{int(r['tp']):>12,}{r['oni_aso']:>+10.2f}"
          f"{r['dmi_aso']:>+10.2f}{r['oni_mjj']:>+10.2f}{r['dmi_mjj']:>+10.2f}")


def r2(X, y):
    m = LinearRegression().fit(X, y)
    return m.score(X, y), m.coef_


print(f"\nn = {len(T)} tahun")
for nama, kol in [
    ("ONI ASO saja", ["oni_aso"]),
    ("DMI ASO saja", ["dmi_aso"]),
    ("ONI + DMI ASO", ["oni_aso", "dmi_aso"]),
    ("ONI MJJ saja  (dapat diramalkan)", ["oni_mjj"]),
    ("DMI MJJ saja  (dapat diramalkan)", ["dmi_mjj"]),
    ("ONI + DMI MJJ (dapat diramalkan)", ["oni_mjj", "dmi_mjj"]),
]:
    R2, koef = r2(T[kol].to_numpy(), T["tp"].to_numpy())
    k = ", ".join(f"{c}={v:+,.0f}" for c, v in zip(kol, koef))
    print(f"  {nama:<34} R2 = {R2:.3f}   ({k})")

kor = T[["tp", "oni_aso", "dmi_aso", "oni_mjj", "dmi_mjj"]].corr()["tp"]
print("\n  korelasi Pearson dengan titik panas tahunan:")
for k in ["oni_aso", "dmi_aso", "oni_mjj", "dmi_mjj"]:
    print(f"    {k:<10} r = {kor[k]:+.3f}")

# --------------------------------------------------------------- B. ambang

print("\n" + "=" * 74)
print("B. SENSITIVITAS AMBANG — apakah temuan bertahan?")
print("=" * 74)

FITUR_D = FITUR + ["dmi_lag1", "dmi_lag2", "dmi_lag3"]


def siapkan(panel_path, oni_path, kuantil):
    """Ambang dihitung pada kerangka SEBELUM dropna, persis seperti muat(),
    supaya baris persentil 90 mereproduksi angka yang sudah diterbitkan."""
    d, _ = muat(panel_path, oni_path)
    raw = pd.read_csv(panel_path)
    raw = raw[raw["tahun"] <= 2025]
    amb = raw["titik_panas"].quantile(kuantil)
    for lag in (1, 2, 3):
        d[f"dmi_lag{lag}"] = (d["periode"] - lag).map(lambda p: dmi.get(str(p), np.nan))
    d["y"] = (d["titik_panas"] > amb).astype(int)
    return d, amb


def evaluasi(d):
    kum = {}
    for th in range(2019, 2026):
        latih, uji = d[d["tahun"] < th], d[d["tahun"] == th]
        skor = skor_model(latih, uji)
        Xl = latih[FITUR_D].fillna(0.0).to_numpy()
        Xu = uji[FITUR_D].fillna(0.0).to_numpy()
        xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                            eval_metric="logloss", random_state=42, verbosity=0)
        xgb.fit(Xl, latih["y"].to_numpy())
        skor["gb+DMI"] = (xgb.predict_proba(Xl)[:, 1], xgb.predict_proba(Xu)[:, 1])
        for n, (sl, su) in skor.items():
            kum.setdefault(n, {"y": [], "raw": []})
            kum[n]["y"].append(uji["y"].to_numpy())
            kum[n]["raw"].append(su)
    out = {}
    for n in kum:
        y = np.concatenate(kum[n]["y"]); raw = np.concatenate(kum[n]["raw"])
        out[n] = {"pr": average_precision_score(y, raw), "y": y, "raw": raw}
    return out


def pembalikan(hasil, nama, rng, undian=300):
    """Hitung pasangan berbalik antara peringkat AUC-PR prevalensi asli dan
    peringkat ROC-AUC pada data uji yang diseimbangkan."""
    y = hasil[nama[0]]["y"]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ur_o = sorted(nama, key=lambda n: -hasil[n]["pr"])
    roc_rerata = {n: [] for n in nama}
    balik = []
    for _ in range(undian):
        idx = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        roc = {n: roc_auc_score(y[idx], hasil[n]["raw"][idx]) for n in nama}
        for n in nama:
            roc_rerata[n].append(roc[n])
        ur_l = sorted(nama, key=lambda n: -roc[n])
        c = 0
        for i in range(len(nama)):
            for j in range(i + 1, len(nama)):
                a, b = nama[i], nama[j]
                if (ur_l.index(a) - ur_l.index(b)) * (ur_o.index(a) - ur_o.index(b)) < 0:
                    c += 1
        balik.append(c)
    return np.array(balik), ur_o, {n: float(np.mean(v)) for n, v in roc_rerata.items()}


PAPER = ["klimatologi", "persistence", "seasonal-naive", "rasio-analog",
         "regresi-ONI", "regresi-penuh", "gradient-boosting"]

ringkas = []
for kuantil in (0.80, 0.90, 0.95):
    rng = np.random.default_rng(7)
    d, amb = siapkan(panel_path, oni_path, kuantil)
    hasil = evaluasi(d)
    nama = [n for n in PAPER if n in hasil]          # tujuh model artikel saja
    balik, ur_o, roc_r = pembalikan(hasil, nama, rng)
    npair = len(nama) * (len(nama) - 1) // 2
    prev = hasil[nama[0]]["y"].mean()
    terbaik_roc = max(roc_r, key=roc_r.get)
    print(f"\n--- persentil {int(kuantil * 100)}: ambang > {amb:.0f}, "
          f"prevalensi uji {prev * 100:.1f}% ---")
    print(f"  pembalikan pasangan : median {int(np.median(balik))} dari {npair}, "
          f"rentang {balik.min()}–{balik.max()}, "
          f"nol pembalikan pada {(balik == 0).mean() * 100:.0f}% undian")
    print(f"  teratas AUC-PR      : {ur_o[0]} ({hasil[ur_o[0]]['pr']:.3f})")
    print(f"  teratas ROC-AUC     : {terbaik_roc} ({roc_r[terbaik_roc]:.3f})")
    print(f"  sepakat?            : {'ya' if ur_o[0] == terbaik_roc else 'TIDAK'}")
    print("  AUC-PR: " + "  ".join(f"{n}={hasil[n]['pr']:.3f}" for n in ur_o))
    print(f"  gb tanpa DMI = {hasil['gradient-boosting']['pr']:.3f}   "
          f"gb + DMI = {hasil['gb+DMI']['pr']:.3f}   "
          f"selisih = {hasil['gb+DMI']['pr'] - hasil['gradient-boosting']['pr']:+.3f}")
    ringkas.append({
        "persentil": int(kuantil * 100), "ambang": int(round(amb)), "prevalensi": prev,
        "balik_med": int(np.median(balik)), "balik_min": int(balik.min()),
        "balik_max": int(balik.max()), "npair": npair,
        "top_pr": ur_o[0], "top_roc": terbaik_roc,
        "gb": hasil["gradient-boosting"]["pr"], "gb_dmi": hasil["gb+DMI"]["pr"],
        "klim": hasil["klimatologi"]["pr"],
    })

print("\n" + "=" * 74)
print("RINGKASAN untuk tabel naskah")
print("=" * 74)
print(f"{'p':<5}{'ambang':>8}{'prev':>8}{'balik':>12}{'top AUC-PR':>20}{'top ROC':>20}")
for r in ringkas:
    print(f"{r['persentil']:<5}{r['ambang']:>8}{r['prevalensi'] * 100:>7.1f}%"
          f"{str(r['balik_med']) + ' (' + str(r['balik_min']) + '–' + str(r['balik_max']) + ')':>12}"
          f"{r['top_pr']:>20}{r['top_roc']:>20}")

# --------------------------------------------------------------- C. kolinearitas

print("\n" + "=" * 74)
print("C. KOLINEARITAS ONI–DMI — kenapa koefisien gabungan berbalik tanda")
print("=" * 74)
print(f"  korelasi ONI ASO – DMI ASO : r = {T['oni_aso'].corr(T['dmi_aso']):+.3f}")
print(f"  korelasi ONI MJJ – DMI MJJ : r = {T['oni_mjj'].corr(T['dmi_mjj']):+.3f}")
for nama, kol in [("ASO", ["oni_aso", "dmi_aso"]), ("MJJ", ["oni_mjj", "dmi_mjj"])]:
    tunggal = LinearRegression().fit(T[[kol[1]]], T["tp"]).coef_[0]
    gabung = LinearRegression().fit(T[kol], T["tp"]).coef_[1]
    print(f"  koefisien DMI {nama}: sendiri {tunggal:+,.0f}  bersama ONI {gabung:+,.0f}"
          f"  {'BERBALIK TANDA' if tunggal * gabung < 0 else 'tanda tetap'}")
n = len(T)
print("\n  R2 terkoreksi (n = 13, hukuman untuk jumlah prediktor):")
for nama, R2, p in [("ONI ASO", 0.496, 1), ("ONI+DMI ASO", 0.528, 2),
                    ("ONI MJJ", 0.410, 1), ("ONI+DMI MJJ", 0.554, 2)]:
    print(f"    {nama:<14} R2 = {R2:.3f}   R2-adj = {1 - (1 - R2) * (n - 1) / (n - p - 1):.3f}")
