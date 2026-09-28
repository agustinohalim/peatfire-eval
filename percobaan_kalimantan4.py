"""
Percobaan Kalimantan 4 — dua analisis terakhir sebelum naskah (Artikel_Kalimantan_Rencana.md bagian F).

  A. Kestabilan peringkat per provinsi, dua sasaran. Model dilatih pada seluruh Kalimantan
     (rolling-origin 2019-2025) dan dinilai per provinsi. Untuk 1.000 bootstrap baris
     kabupaten-bulan dalam provinsi: seberapa sering tiap model juara AUC-PR, dan selang tau
     peringkat provinsi terhadap peringkat seluruh Kalimantan.

  B. Uji mekanisme. Dugaan Hasil_Kalimantan_2/3: di bawah ambang gabungan, laju dasar tiap
     provinsi masuk ke label, sehingga model yang dilatih pada seluruh Kalimantan (ID) belajar
     membedakan wilayah, bukan musim, dan kalah atau menang terhadap pelatihan lokal (LOKAL)
     menurut provinsi. Bila dugaan benar, memberi model ID informasi wilayah akan menutup
     selisihnya dengan LOKAL. Dua varian:
       ID+PROV : fitur satu-panas provinsi
       ID+KAB  : log(1 + rerata titik panas kabupaten itu pada tahun-tahun latih)
     Dibandingkan dengan ID dan LOKAL dari Hasil_Kalimantan_3.

Kalimantan Utara dilaporkan terpisah dengan catatan (11 positif di ambang gabungan).

Pakai:
    python percobaan_kalimantan4.py DL_FIRE_NASIONAL/panel_kalimantan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py"), encoding="utf-8").read().split("if __name__")[0])

jalur_kal, oni_path = sys.argv[1], sys.argv[2]
df, ambang = muat(jalur_kal, oni_path)
TAHUN_UJI = list(range(2019, 2026))
PROV = sorted(df["provinsi"].unique())
GB, KL = "gradient-boosting", "klimatologi"
LABEL = {"klimatologi": "Climatology", "gradient-boosting": "Gradient boosting",
         "persistence": "Persistence", "seasonal-naive": "Seasonal naive",
         "rasio-analog": "Ratio scaling", "regresi-ONI": "Logistic, ONI",
         "regresi-penuh": "Logistic, full"}
mentah = pd.read_csv(jalur_kal)
q_kab = mentah[mentah["tahun"] <= 2025].groupby("kabupaten")["titik_panas"].quantile(0.90)
d_kab = df.copy()
d_kab["y"] = (d_kab["titik_panas"] > d_kab["kabupaten"].map(q_kab)).astype(int)
SASARAN = [(f"gabungan (> {ambang:.0f})", df), ("per kabupaten", d_kab)]


def peringkat(v):
    u = sorted(v, key=lambda n: -v[n]); return {n: u.index(n) + 1 for n in v}


def xgb_fit_pred(Xl, yl, Xu):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                      random_state=42, verbosity=0)
    m.fit(Xl, yl)
    return m.predict_proba(Xu)[:, 1]


def fitur(d, lalu, varian):
    X = d[FITUR].fillna(0.0).copy()
    if varian == "PROV":
        for p in PROV:
            X["prov_" + p] = (d["provinsi"] == p).astype(float)
    elif varian == "KAB":
        rerata = lalu.groupby("kabupaten")["titik_panas"].mean()
        X["log_rerata_kab"] = np.log1p(d["kabupaten"].map(rerata).fillna(0.0))
    return X.to_numpy()


for judul, d in SASARAN:
    print("\n" + "=" * 96)
    print(f"SASARAN {judul.upper()}")
    print("=" * 96)
    # ---- prediksi ID seluruh model, plus varian GB
    y, meta, sk = [], [], {}
    for th in TAHUN_UJI:
        lalu, uji = d[d["tahun"] < th], d[d["tahun"] == th]
        s = skor_model(lalu, uji)
        for n, (_, su) in s.items():
            sk.setdefault(n, []).append(su)
        for v in ("PROV", "KAB"):
            sk.setdefault("GB+" + v, []).append(
                xgb_fit_pred(fitur(lalu, lalu, v), lalu["y"].to_numpy(), fitur(uji, lalu, v)))
        lok = []
        for p in PROV:
            ul = uji[uji["provinsi"] == p]
            ll = lalu[lalu["provinsi"] == p]
            pr = xgb_fit_pred(ll[FITUR].fillna(0.0).to_numpy(), ll["y"].to_numpy(),
                              ul[FITUR].fillna(0.0).to_numpy()) if ll["y"].nunique() == 2 else np.full(len(ul), np.nan)
            lok.append(pd.Series(pr, index=ul.index))
        sk.setdefault("GB LOKAL", []).append(pd.concat(lok).reindex(uji.index).to_numpy())
        y.append(uji["y"].to_numpy()); meta.append(uji[["provinsi", "kabupaten"]])
    y = np.concatenate(y); meta = pd.concat(meta, ignore_index=True)
    sk = {n: np.concatenate(v) for n, v in sk.items()}
    TUJUH = list(LABEL)
    r_kal = peringkat({n: average_precision_score(y, sk[n]) for n in TUJUH})

    # ---- A. kestabilan peringkat per provinsi
    print(f"\nA. Peringkat per provinsi (model dilatih pada seluruh Kalimantan), 1.000 bootstrap")
    print(f"  {'provinsi':<18}{'pos':>4}  {'juara titik':<18}{'% GB juara':>11}{'% klim juara':>13}"
          f"{'tau vs Kal':>11}   {'95% tau':<16}")
    for p in PROV:
        m = (meta["provinsi"] == p).to_numpy()
        yp = y[m]
        if yp.sum() < 5:
            print(f"  {p:<18}{int(yp.sum()):>4}  terlalu sedikit positif"); continue
        rp = peringkat({n: average_precision_score(yp, sk[n][m]) for n in TUJUH})
        juara = min(rp, key=rp.get)
        rng = np.random.default_rng(11)
        idx = np.where(m)[0]
        taus, juara_b = [], []
        for _ in range(1000):
            i = rng.choice(idx, size=len(idx), replace=True)
            if y[i].sum() == 0:
                continue
            rb = peringkat({n: average_precision_score(y[i], sk[n][i]) for n in TUJUH})
            taus.append(kendalltau([rb[n] for n in TUJUH], [r_kal[n] for n in TUJUH]).statistic)
            juara_b.append(min(rb, key=rb.get))
        juara_b = pd.Series(juara_b)
        t = kendalltau([rp[n] for n in TUJUH], [r_kal[n] for n in TUJUH]).statistic
        lo, hi = np.percentile(taus, [2.5, 97.5])
        catatan = "  (Kaltara: sedikit positif)" if p == "KalimantanUtara" else ""
        print(f"  {p:<18}{int(yp.sum()):>4}  {LABEL[juara]:<18}{(juara_b == GB).mean() * 100:>10.0f}%"
              f"{(juara_b == KL).mean() * 100:>12.0f}%{t:>11.3f}   [{lo:+.3f}, {hi:+.3f}]{catatan}")

    # ---- B. uji mekanisme
    print(f"\nB. Uji mekanisme: GB ID, ID+PROV, ID+KAB, LOKAL (AUC-PR per provinsi)")
    print(f"  {'provinsi':<18}{'ID':>7}{'ID+PROV':>9}{'ID+KAB':>8}{'LOKAL':>7}   "
          f"{'LOKAL-ID':>9}{'LOKAL-(ID+PROV)':>17}{'LOKAL-(ID+KAB)':>16}")
    for p in PROV:
        m = (meta["provinsi"] == p).to_numpy()
        yp = y[m]
        ok = m & ~np.isnan(sk["GB LOKAL"])
        if yp.sum() < 5:
            print(f"  {p:<18}  terlalu sedikit positif"); continue
        a = {k: average_precision_score(y[ok], sk[k][ok]) for k in (GB, "GB+PROV", "GB+KAB", "GB LOKAL")}
        print(f"  {p:<18}{a[GB]:>7.3f}{a['GB+PROV']:>9.3f}{a['GB+KAB']:>8.3f}{a['GB LOKAL']:>7.3f}   "
              f"{a['GB LOKAL'] - a[GB]:>+9.3f}{a['GB LOKAL'] - a['GB+PROV']:>+17.3f}{a['GB LOKAL'] - a['GB+KAB']:>+16.3f}")
    semua = {k: average_precision_score(y, sk[k]) for k in (GB, "GB+PROV", "GB+KAB")}
    print(f"  seluruh Kalimantan: ID {semua[GB]:.3f}, ID+PROV {semua['GB+PROV']:.3f}, ID+KAB {semua['GB+KAB']:.3f}")
print("\nselesai.")
