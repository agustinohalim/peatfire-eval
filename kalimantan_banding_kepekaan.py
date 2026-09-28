"""
Artikel Kalimantan — bandingkan angka utama antara ambang baku (2012-2025) dan ambang dari tahun
latih saja (kalimantan_kepekaan_ambang.py), pada skema rolling 2019-2025, AUC-PR digabung.

  1. GB - klimatologi seluruh Kalimantan dan per provinsi
  2. rugi transfer: GB ID - GB LOPO per provinsi
  3. tau Kendall peringkat tujuh model per provinsi lawan peringkat seluruh Kalimantan
  4. tahun ekstrem (2015, 2014): GB - klimatologi

Pakai: python kalimantan_banding_kepekaan.py [direktori berisi kedua berkas prediksi; baku DL_FIRE_NASIONAL]
"""

import os
import sys

import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score as ap

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
TUJUH = ["klimatologi", "persistence", "seasonal-naive", "rasio-analog", "regresi-ONI", "regresi-penuh",
         "gradient-boosting"]


def ringkas(P):
    out = {}
    for s in ("gabungan", "kabupaten"):
        r = P[(P.sasaran == s) & (P.skema == "rolling")]
        semua = [ap(r.y, r[m]) for m in TUJUH]
        out[(s, "Kalimantan", "GB-klim")] = semua[-1] - semua[0]
        for p, g in r.groupby("provinsi"):
            if g.y.sum() < 5:
                continue
            v = [ap(g.y, g[m]) for m in TUJUH]
            out[(s, p, "GB-klim")] = v[-1] - v[0]
            out[(s, p, "ID-LOPO")] = ap(g.y, g["gradient-boosting"]) - ap(g.y, g["gb_lopo"])
            out[(s, p, "tau")] = kendalltau(v, semua)[0]
        for th in (2015, 2014):
            e = P[(P.sasaran == s) & (P.skema == f"ekstrem_{th}")]
            out[(s, f"ekstrem {th}", "GB-klim")] = ap(e.y, e["gradient-boosting"]) - ap(e.y, e["klimatologi"])
    return pd.Series(out)


a = ringkas(pd.read_csv(os.path.join(D, "prediksi_kalimantan.csv")))
b = ringkas(pd.read_csv(os.path.join(D, "prediksi_kalimantan_kepekaan.csv")))
t = pd.DataFrame({"baku": a, "ambang_latih": b})
t["selisih"] = t.ambang_latih - t.baku
pd.set_option("display.width", 140)
print(t.round(3).to_string())
