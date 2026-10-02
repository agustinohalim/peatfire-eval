"""
PeatFireBench — kepekaan terhadap benih acak (panel telaah ketiga, Telaah/PeatFireBench_2026-10-02b,
temuan methods:M4 dan klaster temuan 5).

Random forest, gradient boosting, dan MLP dilatih ulang dengan lima benih lain (1-5) pada skema
S1, tahun uji 2020-2025, dua sasaran; fitur, parameter, dan data sama persis dengan
patokan_prediksi.py (definisinya dipakai lewat exec, hanya random_state yang diganti). Baseline
lain tidak bergantung pada benih dan diambil dari patokan_prediksi_<putaran>.csv.

Keluaran: DL_FIRE_NASIONAL/patokan_benih.csv  (kunci baris + rf_b, xgb_b, mlp_b untuk b = 1..5)
Analisis: patokan_benih_analisis.py

Pakai:    python patokan_benih.py     (latar belakang; ± 30-60 menit)
"""

import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SUMBER = open(os.path.join(BASE, "patokan_prediksi.py"), encoding="utf-8").read().split("baris = []")[0]
assert SUMBER.count("random_state=42") == 3
exec(SUMBER.replace("random_state=42", "random_state=BENIH"))

BENIH_LIST = (1, 2, 3, 4, 5)
TAHUN = range(2020, 2026)
kunci = ["gid", "provinsi", "tahun", "bulan", "ada_positif"]
keluar = []
for ylab in ("y_kabupaten", "y_gabungan"):
    for th in TAHUN:
        latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
        b = uji[kunci].copy()
        b["y"], b["sasaran"], b["skema"], b["tag"] = uji[ylab].to_numpy(), ylab, "S1", str(th)
        for BENIH in BENIH_LIST:
            s = dipelajari(latih, uji, ylab)
            for m in ("rf", "xgb", "mlp"):
                b[f"{m}_{BENIH}"] = s[m]
        keluar.append(b)
        print(f"{ylab} {th}", flush=True)
out = pd.concat(keluar, ignore_index=True)
out.to_csv(os.path.join(D, "patokan_benih.csv"), index=False)
print(f"tersimpan: patokan_benih.csv  baris: {len(out)}")
