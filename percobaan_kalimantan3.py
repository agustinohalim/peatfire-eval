"""
Percobaan Kalimantan 3 — tiga pemeriksaan sebelum temuan Hasil_Kalimantan_2 diklaim.

  A. Selang bootstrap selisih AUC-PR gradient boosting LOKAL (dilatih pada provinsi sendiri)
     lawan ID (dilatih pada seluruh Kalimantan), per provinsi, dua definisi sasaran. Dua skema:
     baris kabupaten-bulan (iid) dan klaster kabupaten.
  B. Regresi logistik dengan fitur diskalakan (StandardScaler). Peringatan konvergensi di
     Hasil_Kalimantan_1 berasal dari fitur mentah; di sini dibandingkan peringkat tujuh model
     dengan logistik mentah lawan logistik terskala. `percobaan.py` TIDAK diubah, karena angka
     Artikel 1 bergantung padanya.
  C. Tahun ekstrem: dua tahun dengan deteksi terbanyak di Kalimantan disisihkan penuh dari
     pelatihan dan dipakai sebagai tahun uji, seperti Artikel 1 Bagian 4.5. Skill ternormalisasi
     GB lawan klimatologi per provinsi.

Pakai:
    python percobaan_kalimantan3.py DL_FIRE_NASIONAL/panel_kalimantan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py"), encoding="utf-8").read().split("if __name__")[0])

jalur_kal, oni_path = sys.argv[1], sys.argv[2]
df, ambang = muat(jalur_kal, oni_path)
TAHUN_UJI = list(range(2019, 2026))
PROV = sorted(df["provinsi"].unique())
GB, KL = "gradient-boosting", "klimatologi"
mentah = pd.read_csv(jalur_kal)
q_kab = mentah[mentah["tahun"] <= 2025].groupby("kabupaten")["titik_panas"].quantile(0.90)
d_kab = df.copy()
d_kab["y"] = (d_kab["titik_panas"] > d_kab["kabupaten"].map(q_kab)).astype(int)
SASARAN = [(f"gabungan (> {ambang:.0f})", df), ("per kabupaten", d_kab)]


def gb(latih, uji):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                      random_state=42, verbosity=0)
    m.fit(latih[FITUR].fillna(0.0).to_numpy(), latih["y"].to_numpy())
    return m.predict_proba(uji[FITUR].fillna(0.0).to_numpy())[:, 1]


def skill(y, s):
    p = y.mean()
    return (average_precision_score(y, s) - p) / (1 - p)


# ============================================================ A. LOKAL lawan ID, bootstrap
print("=" * 90)
print("A. GB DILATIH PADA PROVINSI SENDIRI (LOKAL) LAWAN SELURUH KALIMANTAN (ID)")
print("=" * 90)
for judul, d in SASARAN:
    print(f"\n  sasaran {judul}")
    print(f"  {'provinsi':<18}{'pos':>4}{'ID':>7}{'LOKAL':>7}{'LOKAL-ID':>9}   {'iid 95%':<18}{'klaster kab 95%':<18}")
    for p in PROV:
        ys, sid, slok, kab = [], [], [], []
        for th in TAHUN_UJI:
            lalu = d[d["tahun"] < th]
            uji = d[(d["tahun"] == th) & (d["provinsi"] == p)]
            lok = lalu[lalu["provinsi"] == p]
            if lok["y"].nunique() < 2:
                continue
            ys.append(uji["y"].to_numpy()); sid.append(gb(lalu, uji)); slok.append(gb(lok, uji))
            kab.append(uji["kabupaten"].to_numpy())
        y, sid, slok, kab = map(np.concatenate, (ys, sid, slok, kab))
        if y.sum() < 5:
            print(f"  {p:<18}{int(y.sum()):>4}  terlalu sedikit positif"); continue
        a_id, a_lok = average_precision_score(y, sid), average_precision_score(y, slok)
        rng = np.random.default_rng(2026)
        iid, kl = [], []
        uk = np.unique(kab); idx = {k: np.where(kab == k)[0] for k in uk}
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum():
                iid.append(average_precision_score(y[i], slok[i]) - average_precision_score(y[i], sid[i]))
            j = np.concatenate([idx[k] for k in rng.choice(uk, size=len(uk), replace=True)])
            if y[j].sum():
                kl.append(average_precision_score(y[j], slok[j]) - average_precision_score(y[j], sid[j]))
        a, b = np.percentile(iid, [2.5, 97.5]); c, e = np.percentile(kl, [2.5, 97.5])
        print(f"  {p:<18}{int(y.sum()):>4}{a_id:>7.3f}{a_lok:>7.3f}{a_lok - a_id:>+9.3f}   "
              f"[{a:+.3f}, {b:+.3f}]  [{c:+.3f}, {e:+.3f}]")

# ============================================================ B. logistik terskala
print("\n" + "=" * 90)
print("B. REGRESI LOGISTIK MENTAH LAWAN TERSKALA (seluruh Kalimantan, rolling-origin)")
print("=" * 90)
ONI_COLS = [f"oni_lag{l}" for l in range(1, 7)] + ["bulan_sin", "bulan_cos"]


def logistik(latih, uji, kol, skala):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if skala \
        else LogisticRegression(max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(latih[kol].fillna(0.0), latih["y"])
    return m.predict_proba(uji[kol].fillna(0.0))[:, 1]


def peringkat(v):
    u = sorted(v, key=lambda n: -v[n]); return {n: u.index(n) + 1 for n in v}


for judul, d in SASARAN:
    y, sk = [], {}
    for th in TAHUN_UJI:
        latih, uji = d[d["tahun"] < th], d[d["tahun"] == th]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = skor_model(latih, uji)
        for n, (_, su) in s.items():
            sk.setdefault(n, []).append(su)
        for nama, kol in (("regresi-ONI", ONI_COLS), ("regresi-penuh", FITUR)):
            sk.setdefault(nama + " (skala)", []).append(logistik(latih, uji, kol, True))
        y.append(uji["y"].to_numpy())
    y = np.concatenate(y); sk = {n: np.concatenate(v) for n, v in sk.items()}
    mentah_n = [n for n in sk if "skala" not in n]
    skala_n = [n for n in mentah_n if not n.startswith("regresi")] + ["regresi-ONI (skala)", "regresi-penuh (skala)"]
    print(f"\n  sasaran {judul}")
    for n in ("regresi-ONI", "regresi-penuh"):
        print(f"    {n:<14} AUC-PR mentah {average_precision_score(y, sk[n]):.3f}  terskala "
              f"{average_precision_score(y, sk[n + ' (skala)']):.3f}   ROC-AUC mentah "
              f"{roc_auc_score(y, sk[n]):.3f}  terskala {roc_auc_score(y, sk[n + ' (skala)']):.3f}")
    for label, nama in (("mentah", mentah_n), ("terskala", skala_n)):
        C = peringkat({n: roc_auc_score(y, sk[n]) for n in nama})
        D = peringkat({n: average_precision_score(y, sk[n]) for n in nama})
        k = list(C)
        t = kendalltau([C[n] for n in k], [D[n] for n in k]).statistic
        urut = " > ".join(n.replace(" (skala)", "") for n in sorted(D, key=D.get))
        print(f"    {label:<9} tau ROC lawan PR = {t:+.3f}   urutan AUC-PR: {urut}")

# ============================================================ C. tahun ekstrem
print("\n" + "=" * 90)
besar = df.groupby("tahun")["titik_panas"].sum().sort_values(ascending=False)
ekstrem = list(besar.index[:2])
print(f"C. TAHUN EKSTREM DISISIHKAN: {ekstrem} (deteksi {', '.join(f'{int(besar[t]):,}' for t in ekstrem)})")
print("=" * 90)
for judul, d in SASARAN:
    print(f"\n  sasaran {judul}")
    print(f"  {'tahun':<6}{'provinsi':<18}{'pos':>4}{'prev':>7}{'GB':>7}{'klim':>7}{'GB-klim':>9}   95%")
    for th in ekstrem:
        sisa, uji = d[d["tahun"] != th], d[d["tahun"] == th]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = skor_model(sisa, uji)
        for p in PROV:
            m = (uji["provinsi"] == p).to_numpy()
            y = uji["y"].to_numpy()[m]
            if y.sum() < 3 or y.sum() == len(y):
                print(f"  {th:<6}{p:<18}{int(y.sum()):>4}  tidak dapat dinilai"); continue
            g, k = s[GB][1][m], s[KL][1][m]
            rng = np.random.default_rng(7); bb = []
            for _ in range(2000):
                i = rng.integers(0, len(y), len(y))
                if 0 < y[i].sum() < len(i):
                    bb.append(skill(y[i], g[i]) - skill(y[i], k[i]))
            lo, hi = np.percentile(bb, [2.5, 97.5])
            print(f"  {th:<6}{p:<18}{int(y.sum()):>4}{y.mean() * 100:>6.1f}%{skill(y, g):>7.3f}{skill(y, k):>7.3f}"
                  f"{skill(y, g) - skill(y, k):>+9.3f}   [{lo:+.3f}, {hi:+.3f}]")
print("\nselesai.")
