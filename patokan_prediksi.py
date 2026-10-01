"""
Artikel 2 (PeatFireBench) — prediksi seluruh baseline tolok ukur nasional, disimpan sekali.

Data   : DL_FIRE_NASIONAL/panel_nasional_fitur_bersih.csv (498 kabupaten/kota, 2013-2025,
         label y_kabupaten = > maks(p90 kabupaten, 10) dan y_gabungan = > p90 nasional).
Kunci  : gid (GADM), bukan nama; "Banjar" dipakai dua kabupaten berbeda.

Model (Artikel_2_Rencana bagian D; model 8 FWI ditunda sampai datanya ada):
  klimatologi, persistence, seasonal_naive, rasio   -- baseline lokal, dari sejarah kabupaten
  logistik  : regresi logistik terskala pada lag ONI 1-6, lag DMI 1-3, suku bulan
  rf        : random forest pada seluruh fitur
  xgb       : gradient boosting (parameter sama dengan Artikel 1)
  mlp       : jaringan saraf ringkas (satu lapis tersembunyi 16 unit, sklearn), setara ProbFire

Skema pembagian:
  S1 kronologis   : uji 2019-2025, latih tahun-tahun sebelumnya (rolling-origin)
  S2 tahun ekstrem: 2015, 2014, 2019 (tiga tahun deteksi terbanyak nasional) masing-masing
                    disisihkan penuh; latih pada semua tahun lain
  S3 transfer     : lima kelompok pulau; untuk tiap tahun uji 2019-2025 dan tiap kelompok, model
                    yang dipelajari dilatih pada empat kelompok lain (tahun < uji); baseline lokal
                    tetap dari sejarah kabupaten uji sendiri

Pakai:
    python patokan_prediksi.py            (± 30-60 menit; jalankan di latar belakang)
    PATOKAN_FITUR=hujan|lengkap python patokan_prediksi.py   (lihat kumpulan fitur di bawah)

Keluaran: DL_FIRE_NASIONAL/patokan_prediksi.csv (tidak masuk repo)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
# Kumpulan fitur untuk rf, xgb, dan mlp; logistik tetap iklim global saja di semua putaran.
#   PATOKAN_FITUR=dasar   (bawaan)  lag titik panas, ONI, DMI, musim
#   PATOKAN_FITUR=hujan             + jeda hujan CHIRPS (gabung_chirps_nasional.py)
#   PATOKAN_FITUR=lengkap           + lapisan lahan dan manusia (gabung_lahan_nasional.py)
# PATOKAN_HUJAN=1 dari putaran 25 September 2026 tetap berarti "hujan".
from kelompok_fitur import KELOMPOK
FITUR_SET = os.environ.get("PATOKAN_FITUR") or ("hujan" if os.environ.get("PATOKAN_HUJAN") == "1" else "dasar")
BERKAS = {"dasar": ("panel_nasional_fitur_bersih.csv", "patokan_prediksi.csv"),
          "hujan": ("panel_nasional_fitur_hujan.csv", "patokan_prediksi_hujan.csv"),
          "lengkap": ("panel_nasional_fitur_lengkap.csv", "patokan_prediksi_lengkap.csv")}[FITUR_SET]
OUT = os.path.join(D, BERKAS[1])

df = pd.read_csv(os.path.join(D, BERKAS[0]))
FITUR = KELOMPOK["api"] + KELOMPOK["musim"] + KELOMPOK["iklim"]
if FITUR_SET in ("hujan", "lengkap"):
    FITUR += KELOMPOK["hujan"]
if FITUR_SET == "lengkap":
    FITUR += KELOMPOK["lahan"] + KELOMPOK["manusia"]
IKLIM = [f"oni_lag{l}" for l in range(1, 7)] + [f"dmi_lag{l}" for l in range(1, 4)] + ["bulan_sin", "bulan_cos"]
DIPELAJARI = ["logistik", "rf", "xgb", "mlp"]
PULAU = {
    "Sumatera": ["Aceh", "SumateraUtara", "SumateraBarat", "Riau", "KepulauanRiau", "Jambi",
                 "SumateraSelatan", "Bengkulu", "Lampung", "BangkaBelitung"],
    "Kalimantan": ["KalimantanBarat", "KalimantanTengah", "KalimantanSelatan", "KalimantanTimur",
                   "KalimantanUtara"],
    "Jawa-Bali-NT": ["Banten", "JakartaRaya", "JawaBarat", "JawaTengah", "Yogyakarta", "JawaTimur",
                     "Bali", "NusaTenggaraBarat", "NusaTenggaraTimur"],
    "Sulawesi": ["SulawesiUtara", "Gorontalo", "SulawesiTengah", "SulawesiBarat", "SulawesiSelatan",
                 "SulawesiTenggara"],
    "Maluku-Papua": ["Maluku", "MalukuUtara", "Papua", "PapuaBarat"],
}
KE_PULAU = {p: k for k, v in PULAU.items() for p in v}
assert set(df["provinsi"]) <= set(KE_PULAU), set(df["provinsi"]) - set(KE_PULAU)
df["pulau"] = df["provinsi"].map(KE_PULAU)


def lokal(latih, uji):
    """Baseline lokal dari sejarah kabupaten di data latih (kunci gid)."""
    klim = latih.groupby(["gid", "bulan_ke"])["titik_panas"].mean()
    idx = pd.MultiIndex.from_arrays([uji["gid"], uji["bulan_ke"]])
    k = klim.reindex(idx).fillna(0.0).to_numpy()
    klim_l = latih.groupby(["gid", "bulan_ke"])["titik_panas"].mean()
    base = k
    lalu = uji[["tp_lag1", "tp_lag2"]].fillna(0.0).sum(axis=1).to_numpy()
    rasio = base * np.clip(lalu / np.maximum(base * 2, 1e-9), 0.2, 5.0)
    return {"klimatologi": k, "persistence": uji["tp_lag1"].fillna(0.0).to_numpy(),
            "seasonal_naive": uji["tp_lag12"].fillna(0.0).to_numpy(), "rasio": rasio}


def dipelajari(latih, uji, ylab):
    Xl, Xu = latih[FITUR].fillna(0.0).to_numpy(), uji[FITUR].fillna(0.0).to_numpy()
    yl = latih[ylab].to_numpy()
    out = {}
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    lg.fit(latih[IKLIM].fillna(0.0), yl)
    out["logistik"] = lg.predict_proba(uji[IKLIM].fillna(0.0))[:, 1]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(Xl, yl)
    out["rf"] = rf.predict_proba(Xu)[:, 1]
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                       colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                       random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(Xl, yl)
    out["xgb"] = xg.predict_proba(Xu)[:, 1]
    # lag titik panas dilog-kan untuk MLP supaya skala antar kabupaten sebanding
    def tr(d):
        x = d[FITUR].fillna(0.0).copy()
        for c in ("tp_lag1", "tp_lag2", "tp_lag3", "tp_lag12"):
            x[c] = np.log1p(x[c])
        return x.to_numpy()
    ml = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), max_iter=300,
                                                       early_stopping=True, random_state=42))
    ml.fit(tr(latih), yl)
    out["mlp"] = ml.predict_proba(tr(uji))[:, 1]
    return out


baris = []


def simpan(uji, skema, ylab, tag, skor):
    b = uji[["gid", "provinsi", "pulau", "tahun", "bulan", "ada_positif"]].copy()
    b["y"] = uji[ylab].to_numpy()
    b["sasaran"], b["skema"], b["tag"] = ylab, skema, tag
    for n, v in skor.items():
        b[n] = v
    baris.append(b)


for ylab in ("y_kabupaten", "y_gabungan"):
    # S1 kronologis
    for th in range(2019, 2026):
        latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
        simpan(uji, "S1", ylab, str(th), {**lokal(latih, uji), **dipelajari(latih, uji, ylab)})
        print(f"{ylab} S1 {th}", flush=True)
    # S2 tahun ekstrem
    for th in (2015, 2014, 2019):
        latih, uji = df[df["tahun"] != th], df[df["tahun"] == th]
        simpan(uji, "S2", ylab, str(th), {**lokal(latih, uji), **dipelajari(latih, uji, ylab)})
        print(f"{ylab} S2 {th}", flush=True)
    # S3 transfer antarpulau
    for th in range(2019, 2026):
        lalu = df[df["tahun"] < th]
        for g in PULAU:
            uji = df[(df["tahun"] == th) & (df["pulau"] == g)]
            skor = {**lokal(lalu, uji), **dipelajari(lalu[lalu["pulau"] != g], uji, ylab)}
            simpan(uji, "S3", ylab, f"{th}_{g}", skor)
        print(f"{ylab} S3 {th}", flush=True)

hasil = pd.concat(baris, ignore_index=True)
hasil.to_csv(OUT, index=False)
print(f"tersimpan: {OUT}  baris: {len(hasil)}")
