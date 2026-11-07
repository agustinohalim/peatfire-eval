"""
PeatFireBench revisi fase A — baseline klimatologi frekuensi lampau dan skema Q (Hasil_Patokan_6).

Baseline baru, dari tahun latih saja:
  klim_frek     bagian kabupaten-bulan latih yang positif dalam jendela tiga bulan kalender
                (m-1, m, m+1, melingkar) di kabupaten itu; seri dipecah 1e-6 * k/(1+k),
                k = klimatologi rata-rata hitungan. Rujukan utama dan rujukan Brier.
  klim_frek1    sama tanpa jendela (bulan kalender itu saja).
  log_klim_jeda regresi logistik terskala: log1p(klimatologi), log1p(tp_lag1..3), bulan_sin/cos.
                Klimatologi baris latih dihitung tanpa tahun baris itu sendiri.

Skema:
  S1, S2  baseline baru saja; kolom model lama digabung dari patokan_prediksi_penuh.csv apa adanya.
  Q       tahun 2020-2025 masing-masing disisihkan, latih pada semua tahun lain; rf, xgb, mlp
          dilatih ulang (parameter sama dengan patokan_prediksi.py), plus klimatologi,
          persistence, dan baseline baru.

Pakai:    python patokan_revisi_prediksi.py   (log ke DL_FIRE_NASIONAL/penuh/patokan_revisi_prediksi.log)
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_prediksi.csv (tidak masuk repo)
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

from kelompok_fitur import KELOMPOK, PUTARAN, PREDIKSI, PANEL

warnings.filterwarnings("ignore")
assert PUTARAN == "penuh", PUTARAN
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
OUT = os.path.join(D, "patokan_revisi_prediksi.csv")
df = pd.read_csv(os.path.join(D, PANEL))
# urutan fitur sama dengan patokan_prediksi.py putaran penuh
FITUR = (KELOMPOK["api"] + KELOMPOK["musim"] + KELOMPOK["iklim"] + KELOMPOK["hujan"] + KELOMPOK["lahan"]
         + KELOMPOK["manusia"] + KELOMPOK["gambut"] + KELOMPOK["cuaca"])
SAS = ("y_kabupaten", "y_gabungan")
KUNCI = ["sasaran", "skema", "tag", "gid", "tahun", "bulan"]


def tabel_klim(latih, kol):
    """jumlah dan banyak tahun per (gid, bulan_ke) di data latih, lengkap 12 bulan."""
    g = latih.groupby(["gid", "bulan_ke"])[kol].agg(["sum", "count"])
    idx = pd.MultiIndex.from_product([latih["gid"].unique(), range(1, 13)], names=["gid", "bulan_ke"])
    return g.reindex(idx, fill_value=0)


def ambil(tab, uji, jendela):
    """nilai rata-rata (sum/count) untuk baris uji, dijumlah dalam jendela bulan kalender melingkar."""
    s = np.zeros(len(uji)); n = np.zeros(len(uji))
    for geser in range(-jendela, jendela + 1):
        bk = (uji["bulan_ke"].to_numpy() - 1 + geser) % 12 + 1
        idx = pd.MultiIndex.from_arrays([uji["gid"].to_numpy(), bk])
        t = tab.reindex(idx)
        s += t["sum"].fillna(0).to_numpy(); n += t["count"].fillna(0).to_numpy()
    return s, n


def klim_rata(latih, uji):
    s, n = ambil(tabel_klim(latih, "titik_panas"), uji, 0)
    return np.where(n > 0, s / np.maximum(n, 1), 0.0)


def klim_frek(latih, uji, ylab, jendela):
    s, n = ambil(tabel_klim(latih, ylab), uji, jendela)
    f = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    k = klim_rata(latih, uji)
    return f + 1e-6 * k / (1 + k)


def klim_tanpa_tahun_sendiri(latih):
    """klimatologi rata-rata untuk baris latih, tanpa tahun baris itu (satu baris per gid-bulan-tahun)."""
    tab = tabel_klim(latih, "titik_panas")
    s, n = ambil(tab, latih, 0)
    x = latih["titik_panas"].to_numpy()
    return np.where(n > 1, (s - x) / np.maximum(n - 1, 1), 0.0)


def X_jeda(d, k):
    x = np.column_stack([np.log1p(k)] + [np.log1p(d[c].fillna(0.0).to_numpy()) for c in ("tp_lag1", "tp_lag2", "tp_lag3")]
                        + [d["bulan_sin"].to_numpy(), d["bulan_cos"].to_numpy()])
    return x


def baseline_baru(latih, uji, ylab):
    out = {"klim_frek": klim_frek(latih, uji, ylab, 1), "klim_frek1": klim_frek(latih, uji, ylab, 0)}
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    lg.fit(X_jeda(latih, klim_tanpa_tahun_sendiri(latih)), latih[ylab].to_numpy())
    out["log_klim_jeda"] = lg.predict_proba(X_jeda(uji, klim_rata(latih, uji)))[:, 1]
    return out


def dipelajari(latih, uji, ylab):
    Xl, Xu = latih[FITUR].fillna(0.0).to_numpy(), uji[FITUR].fillna(0.0).to_numpy()
    yl = latih[ylab].to_numpy()
    out = {}
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(Xl, yl); out["rf"] = rf.predict_proba(Xu)[:, 1]
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                       colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                       random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(Xl, yl); out["xgb"] = xg.predict_proba(Xu)[:, 1]

    def tr(d):
        x = d[FITUR].fillna(0.0).copy()
        for c in ("tp_lag1", "tp_lag2", "tp_lag3", "tp_lag12"):
            x[c] = np.log1p(x[c])
        return x.to_numpy()
    ml = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), max_iter=300,
                                                       early_stopping=True, random_state=42))
    ml.fit(tr(latih), yl); out["mlp"] = ml.predict_proba(tr(uji))[:, 1]
    return out


def bingkai(uji, ylab, skema, tag, skor):
    b = uji[["gid", "provinsi", "tahun", "bulan", "ada_positif"]].copy()
    b["y"] = uji[ylab].to_numpy()
    b["sasaran"], b["skema"], b["tag"] = ylab, skema, tag
    for k, v in skor.items():
        b[k] = v
    return b


def dipelajari_xgb(latih, uji, ylab):
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                       colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                       random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(latih[FITUR].fillna(0.0).to_numpy(), latih[ylab].to_numpy())
    return xg.predict_proba(uji[FITUR].fillna(0.0).to_numpy())[:, 1]


def utama():
    lama = pd.read_csv(os.path.join(D, PREDIKSI), dtype={"tag": str})
    lama = lama[lama.skema.isin(["S1", "S2"])]
    baris = []
    for ylab in SAS:
        for th in range(2019, 2026):
            latih, uji = df[df.tahun < th], df[df.tahun == th]
            baris.append(bingkai(uji, ylab, "S1", str(th), {"klimatologi_cek": klim_rata(latih, uji), **baseline_baru(latih, uji, ylab)}))
        for th in (2015, 2014, 2019):
            latih, uji = df[df.tahun != th], df[df.tahun == th]
            baris.append(bingkai(uji, ylab, "S2", str(th), {"klimatologi_cek": klim_rata(latih, uji), **baseline_baru(latih, uji, ylab)}))
        print(f"{ylab}: baseline S1/S2 selesai", flush=True)
    baru = pd.concat(baris, ignore_index=True)
    s12 = lama.merge(baru.drop(columns=["provinsi", "ada_positif", "y"]), on=KUNCI, how="inner", validate="one_to_one")
    assert len(s12) == len(lama) == len(baru), (len(s12), len(lama), len(baru))
    cek = np.abs(s12["klimatologi"] - s12["klimatologi_cek"]).max()
    print(f"cek klimatologi rata-rata dihitung ulang vs berkas lama: selisih maks {cek:.2e}")
    s12 = s12.drop(columns=["klimatologi_cek"])

    baris = []
    for ylab in SAS:
        for th in range(2020, 2026):
            latih, uji = df[df.tahun != th], df[df.tahun == th]
            skor = {"klimatologi": klim_rata(latih, uji), "persistence": uji["tp_lag1"].fillna(0.0).to_numpy(),
                    **baseline_baru(latih, uji, ylab), **dipelajari(latih, uji, ylab)}
            baris.append(bingkai(uji, ylab, "Q", str(th), skor))
            print(f"{ylab} Q {th}", flush=True)
    q = pd.concat(baris, ignore_index=True)

    # cek reproduksi: xgb S2 2015 per kabupaten dilatih ulang
    latih, uji = df[df.tahun != 2015], df[df.tahun == 2015]
    ulang = dipelajari(latih, uji, "y_kabupaten")["xgb"]
    asli = lama[(lama.sasaran == "y_kabupaten") & (lama.skema == "S2") & (lama.tag == "2015")]
    gab = uji[["gid", "bulan"]].assign(ulang=ulang).merge(asli[["gid", "bulan", "xgb"]], on=["gid", "bulan"])
    print(f"cek reproduksi xgb S2 2015 per kabupaten: {len(gab)} baris, selisih maks {np.abs(gab.ulang - gab.xgb).max():.2e}")

    hasil = pd.concat([s12, q], ignore_index=True)
    hasil.to_csv(OUT, index=False)
    print(f"tersimpan: {OUT}  baris {len(hasil)}  (S1/S2 {len(s12)}, Q {len(q)})")


if __name__ == "__main__":
    utama()
