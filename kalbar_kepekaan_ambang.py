"""
Naskah gabungan Kalimantan, bagian 5.1-5.4 (Kalimantan Barat) — uji kepekaan ambang dari tahun latih.

Artikel 1 menghitung ambang sekali dari 2012-2025, termasuk tahun uji. Di sini setiap lipatan
menghitung ulang ambang dari tahun latihnya saja (rolling 2019-2025: 2012 .. th-1; tahun disisihkan
2014, 2015: semua tahun lain) lalu melatih ulang ketujuh model (skor_model dari percobaan.py, benih
sama). Kedua sasaran diuji: ambang gabungan dalam provinsi (p90 semua kabupaten-bulan Kalbar) dan
ambang per kabupaten. Putaran "baku" (ambang 2012-2025) dijalankan dengan kode yang sama sebagai
pemeriksaan reproduksi terhadap angka naskah (AUC-PR GB 0,642, klimatologi 0,611 pada sasaran
gabungan).

Ringkasan per putaran: AUC-PR tiap model (lipatan rolling digabung), peringkat, tau ROC-AUC lawan
AUC-PR pada prevalensi asli, pembalikan pada 300 undian seimbang (median, rentang), selisih GB -
klimatologi [95 %, 2.000 bootstrap kabupaten-bulan], dan per tahun: berapa selisih skill
ternormalisasi yang selangnya tidak memuat nol.

Pakai:    python kalbar_kepekaan_ambang.py DL_FIRE_SV-C2_792597/panel_bulanan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt
Keluaran: layar (simpan ke DL_FIRE_SV-C2_792597/kalbar_kepekaan.log)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import roc_auc_score as auc

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py"), encoding="utf-8").read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
df, _ = muat(panel_path, oni_path)
mentah = pd.read_csv(panel_path)
mentah = mentah[mentah["tahun"] <= 2025]
SEMUA = sorted(mentah["tahun"].unique())
MODEL = ["klimatologi", "persistence", "seasonal-naive", "rasio-analog", "regresi-ONI", "regresi-penuh",
         "gradient-boosting"]
rng = np.random.default_rng(2026)


def label(d, tahun_ambang, sasaran):
    ref = mentah[mentah["tahun"].isin(tahun_ambang)]
    d = d.copy()
    if sasaran == "gabungan":
        d["y"] = (d["titik_panas"] > ref["titik_panas"].quantile(0.90)).astype(int)
    else:
        q = ref.groupby("kabupaten")["titik_panas"].quantile(0.90)
        d["y"] = (d["titik_panas"] > d["kabupaten"].map(q)).astype(int)
    return d


def prediksi(sasaran, baku):
    blok = []
    for th, skema in [(t, "rolling") for t in range(2019, 2026)] + [(2014, "disisihkan"), (2015, "disisihkan")]:
        thn = SEMUA if baku else [t for t in SEMUA if (t < th if skema == "rolling" else t != th)]
        d = label(df, thn, sasaran)
        latih = d[d["tahun"] < th] if skema == "rolling" else d[d["tahun"] != th]
        uji = d[d["tahun"] == th]
        b = uji[["kabupaten", "tahun", "y"]].copy()
        b["skema"] = skema
        for n, (_, s) in skor_model(latih, uji).items():
            b[n] = s
        blok.append(b)
    return pd.concat(blok, ignore_index=True)


def skill(y, s):
    p = y.mean()
    return (ap(y, s) - p) / (1 - p)


def ringkas(P, nama):
    r = P[P.skema == "rolling"]
    y = r.y.to_numpy()
    pr = {m: ap(y, r[m]) for m in MODEL}
    roc = {m: auc(y, r[m]) for m in MODEL}
    urut = sorted(MODEL, key=lambda m: -pr[m])
    tau = kendalltau([pr[m] for m in MODEL], [roc[m] for m in MODEL])[0]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    balik = []
    for _ in range(300):
        i = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        rb = [auc(y[i], r[m].to_numpy()[i]) for m in MODEL]
        balik.append(round(21 * (1 - kendalltau(rb, [pr[m] for m in MODEL])[0]) / 2))
    gb, kl = r["gradient-boosting"].to_numpy(), r["klimatologi"].to_numpy()
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            bs.append(ap(y[i], gb[i]) - ap(y[i], kl[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    tegas = []
    for (sk, th), g in P.groupby(["skema", "tahun"]):
        yy, a, b = g.y.to_numpy(), g["gradient-boosting"].to_numpy(), g["klimatologi"].to_numpy()
        if yy.sum() < 2:
            continue
        v = []
        for _ in range(2000):
            i = rng.integers(0, len(yy), len(yy))
            if yy[i].sum():
                v.append(skill(yy[i], a[i]) - skill(yy[i], b[i]))
        l, h = np.percentile(v, [2.5, 97.5])
        if l > 0 or h < 0:
            tegas.append(f"{th}{'*' if sk == 'disisihkan' else ''}:{'GB' if l > 0 else 'klim'}")
    print(f"\n{nama}: prevalensi uji rolling {y.mean() * 100:.1f}% ({int(y.sum())} positif)")
    print("  AUC-PR  : " + ", ".join(f"{m} {pr[m]:.3f}" for m in urut))
    print(f"  tau ROC-AUC lawan AUC-PR (asli) {tau:.3f} = {round(21 * (1 - tau) / 2)} pasangan sumbang")
    print(f"  pembalikan 300 undian seimbang: median {int(np.median(balik))}, rentang {min(balik)}-{max(balik)}")
    print(f"  GB - klimatologi AUC-PR {pr['gradient-boosting'] - pr['klimatologi']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"  selisih skill per tahun yang selangnya tak memuat nol: {len(tegas)} dari 9 ({', '.join(tegas) or '-'})")


for sasaran in ("gabungan", "kabupaten"):
    for baku in (True, False):
        ringkas(prediksi(sasaran, baku), f"{sasaran.upper()} {'ambang baku 2012-2025' if baku else 'ambang dari tahun latih'}")
