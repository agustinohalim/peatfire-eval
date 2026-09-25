"""
Percobaan 1 dan 2 untuk Artikel 1 — versi Python dengan kalibrasi yang benar.

Percobaan 1: dua protokol evaluasi pada data yang sama.
    Protokol lazim       : data uji DISEIMBANGKAN, metrik ROC-AUC dan akurasi
    Protokol operasional : prevalensi ASLI, metrik AUC-PR, Brier, ECE

Percobaan 2: penyisihan tahun ekstrem — latih tanpa tahun terbesar, uji pada tahun itu.

Perbaikan dibanding versi Node:
    - Kalibrasi regresi isotonik dipasang pada data LATIH, sehingga Brier dan ECE sah
    - Ambang akurasi disetel per model pada kuantil prevalensi data latih, bukan 0,5 asal
    - Model gradient boosting (XGBoost) disertakan

Pakai:
    python percobaan.py DL_FIRE_SV-C2_792597/panel_bulanan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt

Python tidak ada di PATH; jalankan dengan lintasan penuh:
    "$LOCALAPPDATA/Programs/Python/Python312/python.exe" percobaan.py ...
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

RNG = np.random.default_rng(42)
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


# ---------------------------------------------------------------- data

def muat(panel_path, oni_path):
    df = pd.read_csv(panel_path)
    df = df[df["tahun"] <= 2025].copy()
    df["periode"] = pd.PeriodIndex(df["bulan"], freq="M")

    oni_rows = []
    with open(oni_path) as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) < 4 or p[0] not in SEASONS:
                continue
            oni_rows.append((int(p[1]), SEASONS.index(p[0]) + 1, float(p[3])))
    oni = pd.DataFrame(oni_rows, columns=["tahun", "bulan_ke", "oni"])
    oni["periode"] = pd.PeriodIndex(
        oni["tahun"].astype(str) + "-" + oni["bulan_ke"].astype(str).str.zfill(2), freq="M"
    )
    oni_map = dict(zip(oni["periode"], oni["oni"]))

    df = df.sort_values(["kabupaten", "periode"]).reset_index(drop=True)
    g = df.groupby("kabupaten", observed=True)["titik_panas"]
    df["tp_lag1"] = g.shift(1)
    df["tp_lag2"] = g.shift(2)
    df["tp_lag3"] = g.shift(3)
    df["tp_lag12"] = g.shift(12)
    for lag in range(1, 7):
        df[f"oni_lag{lag}"] = (df["periode"] - lag).map(oni_map)
    df["bulan_sin"] = np.sin(2 * np.pi * df["bulan_ke"] / 12)
    df["bulan_cos"] = np.cos(2 * np.pi * df["bulan_ke"] / 12)

    ambang = df["titik_panas"].quantile(0.90)
    df["y"] = (df["titik_panas"] > ambang).astype(int)
    return df.dropna(subset=["tp_lag12"]).reset_index(drop=True), ambang


# ---------------------------------------------------------------- model

FITUR = (
    ["tp_lag1", "tp_lag2", "tp_lag3", "tp_lag12", "bulan_sin", "bulan_cos"]
    + [f"oni_lag{l}" for l in range(1, 7)]
)


def skor_model(latih, uji):
    """Kembalikan dict nama -> (skor_latih, skor_uji). Skor: makin besar makin rawan."""
    out = {}

    klim = latih.groupby(["kabupaten", "bulan_ke"], observed=True)["titik_panas"].mean()

    def amb_klim(d):
        idx = pd.MultiIndex.from_arrays([d["kabupaten"], d["bulan_ke"]])
        return klim.reindex(idx).fillna(0.0).to_numpy()

    out["klimatologi"] = (amb_klim(latih), amb_klim(uji))
    out["persistence"] = (latih["tp_lag1"].to_numpy(), uji["tp_lag1"].to_numpy())
    out["seasonal-naive"] = (latih["tp_lag12"].to_numpy(), uji["tp_lag12"].to_numpy())

    def rasio(d):
        base = amb_klim(d)
        lalu = d[["tp_lag1", "tp_lag2"]].sum(axis=1).to_numpy()
        r = np.divide(lalu, np.maximum(base * 2, 1e-9))
        return base * np.clip(r, 0.2, 5.0)

    out["rasio-analog"] = (rasio(latih), rasio(uji))

    Xl = latih[FITUR].fillna(0.0).to_numpy()
    Xu = uji[FITUR].fillna(0.0).to_numpy()
    yl = latih["y"].to_numpy()

    oni_cols = [f"oni_lag{l}" for l in range(1, 7)] + ["bulan_sin", "bulan_cos"]
    lr_oni = LogisticRegression(max_iter=3000)
    lr_oni.fit(latih[oni_cols].fillna(0.0), yl)
    out["regresi-ONI"] = (
        lr_oni.predict_proba(latih[oni_cols].fillna(0.0))[:, 1],
        lr_oni.predict_proba(uji[oni_cols].fillna(0.0))[:, 1],
    )

    lr_full = LogisticRegression(max_iter=3000)
    lr_full.fit(Xl, yl)
    out["regresi-penuh"] = (lr_full.predict_proba(Xl)[:, 1], lr_full.predict_proba(Xu)[:, 1])

    xgb = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb.fit(Xl, yl)
    out["gradient-boosting"] = (xgb.predict_proba(Xl)[:, 1], xgb.predict_proba(Xu)[:, 1])
    return out


def kalibrator_oof(df, th, nama, tahun_dalam=3):
    """Isotonik per model, dipasang pada prediksi out-of-fold tahun th-3..th-1.

    Tiap tahun dalam diprediksi oleh model yang dilatih pada tahun-tahun sebelumnya. Memasang
    isotonik pada skor data latih sendiri, seperti versi awal, membuat model lentur seperti
    XGBoost tampak terlalu yakin: skor latihnya hampir memisahkan kelas, jadi pemetaan isotonik
    mendorong prediksi uji ke 0 dan 1 (lihat Hasil_Percobaan5.md bagian A).
    """
    s_oof = {n: [] for n in nama}
    y_oof = []
    for dalam in range(th - tahun_dalam, th):
        latih, uji = df[df["tahun"] < dalam], df[df["tahun"] == dalam]
        s = skor_model(latih, uji)
        for n in nama:
            s_oof[n].append(s[n][1])
        y_oof.append(uji["y"].to_numpy())
    y_oof = np.concatenate(y_oof)
    return {n: IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            .fit(np.concatenate(s_oof[n]), y_oof) for n in nama}


# ---------------------------------------------------------------- metrik

def ece(y, p, bins=10):
    tepi = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= tepi[i]) & (p < tepi[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return total


def seimbangkan(idx, y):
    pos = idx[y == 1]
    neg = idx[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return idx
    ambil = RNG.choice(neg, size=min(len(pos), len(neg)), replace=False)
    return np.concatenate([pos, ambil])


# ---------------------------------------------------------------- jalankan

def jalankan(df, tahun_uji_list, label):
    nama = list(skor_model(df[df["tahun"] < min(tahun_uji_list)], df.head(1)).keys())
    kum = {n: {"y": [], "raw": [], "cal": []} for n in nama}

    for uji_th in tahun_uji_list:
        latih = df[df["tahun"] < uji_th]
        uji = df[df["tahun"] == uji_th]
        if len(latih) < 100 or len(uji) == 0:
            continue
        skor = skor_model(latih, uji)
        kal = kalibrator_oof(df, uji_th, nama)
        for n, (sl, su) in skor.items():
            iso = kal[n]
            kum[n]["y"].append(uji["y"].to_numpy())
            kum[n]["raw"].append(su)
            kum[n]["cal"].append(iso.predict(su))

    hasil = []
    for n in nama:
        y = np.concatenate(kum[n]["y"])
        raw = np.concatenate(kum[n]["raw"])
        cal = np.concatenate(kum[n]["cal"])
        idx = np.arange(len(y))
        bidx = seimbangkan(idx, y)

        thr = np.quantile(cal, 1 - y.mean())
        akur_seimbang = ((cal[bidx] >= thr).astype(int) == y[bidx]).mean()

        hasil.append({
            "model": n,
            "lazim_roc": roc_auc_score(y[bidx], raw[bidx]),
            "lazim_akurasi": akur_seimbang,
            "op_pr": average_precision_score(y, raw),
            "op_brier": brier_score_loss(y, cal),
            "op_ece": ece(y, cal),
        })

    h = pd.DataFrame(hasil)
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"n uji = {len(np.concatenate(kum[nama[0]]['y']))}  "
          f"prevalensi asli = {np.concatenate(kum[nama[0]]['y']).mean() * 100:.1f}%")
    print()
    print("                      PROTOKOL LAZIM          PROTOKOL OPERASIONAL")
    print(f"{'model':<20}{'ROC-AUC':>9}{'akurasi':>9}   {'AUC-PR':>8}{'Brier':>9}{'ECE':>8}")
    print("-" * 78)
    for _, r in h.iterrows():
        print(f"{r['model']:<20}{r['lazim_roc']:>9.3f}{r['lazim_akurasi']:>9.3f}   "
              f"{r['op_pr']:>8.3f}{r['op_brier']:>9.4f}{r['op_ece']:>8.4f}")

    ur_l = h.sort_values("lazim_roc", ascending=False)["model"].tolist()
    ur_o = h.sort_values("op_pr", ascending=False)["model"].tolist()
    print(f"\nPeringkat ROC-AUC (lazim)       : {' > '.join(ur_l)}")
    print(f"Peringkat AUC-PR (operasional)  : {' > '.join(ur_o)}")

    balik, total = 0, 0
    for i in range(len(nama)):
        for j in range(i + 1, len(nama)):
            a, b = nama[i], nama[j]
            total += 1
            if (ur_l.index(a) - ur_l.index(b)) * (ur_o.index(a) - ur_o.index(b)) < 0:
                balik += 1
                print(f"  BERBALIK: {a} vs {b}")
    print(f"\n>>> Pasangan berbalik: {balik} dari {total}")
    print(f">>> Terbaik protokol lazim      : {ur_l[0]}")
    print(f">>> Terbaik protokol operasional: {ur_o[0]}")
    return h


if __name__ == "__main__":
    panel_path, oni_path = sys.argv[1], sys.argv[2]
    df, ambang = muat(panel_path, oni_path)
    print(f"Ambang keparahan persentil 90 : > {ambang:.0f} titik panas per kabupaten-bulan")
    print(f"Baris panel dipakai           : {len(df)}")
    print(f"Kabupaten                     : {df['kabupaten'].nunique()}")
    print(f"Rentang                       : {df['bulan'].min()} s/d {df['bulan'].max()}")

    jalankan(df, list(range(2019, 2026)), "PERCOBAAN 1 — rolling-origin, uji 2019-2025")

    besar = df.groupby("tahun", observed=True)["titik_panas"].sum().sort_values(ascending=False)
    print(f"\nTiga tahun terbesar: {list(besar.head(3).index)}")
    for th in besar.head(2).index:
        sisa = df[df["tahun"] != th]
        nama = list(skor_model(sisa, df.head(1)).keys())
        skor = skor_model(sisa, df[df["tahun"] == th])
        y = df[df["tahun"] == th]["y"].to_numpy()
        print(f"\n{'=' * 78}\nPERCOBAAN 2 — sisihkan {th} dari pelatihan, uji pada {th}\n{'=' * 78}")
        print(f"{'model':<20}{'AUC-PR':>9}")
        print("-" * 32)
        for n in nama:
            print(f"{n:<20}{average_precision_score(y, skor[n][1]):>9.3f}")
