"""
PeatFireBench revisi fase A — prakiraan luar-lipatan pada tahun latih (Hasil_Patokan_6, A4-A5).

Sasaran per kabupaten. Untuk tiap lipatan luar (S1 2020-2025: latih tahun < uji; S2 2014, 2015,
2019: latih semua tahun lain) dan tiap tahun latih v: xgb dilatih pada tahun latih tanpa v dan
memprakirakan v; klim_frek, klimatologi, dan log_klim_jeda dihitung dari tahun latih tanpa v.
Ambang keputusan kemudian dipilih pada prakiraan ini, tanpa melihat tahun uji.

Pakai:    python patokan_revisi_oof.py   (log ke DL_FIRE_NASIONAL/penuh/patokan_revisi_oof.log)
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_oof.csv (tidak masuk repo)
"""
import os
import time

import pandas as pd

import patokan_revisi_prediksi as R  # fungsi saja; bagian utama dijaga __main__

YLAB = "y_kabupaten"
LUAR = [("S1", t) for t in range(2020, 2026)] + [("S2", t) for t in (2014, 2015, 2019)]
OUT = os.path.join(R.D, "patokan_revisi_oof.csv")
df = R.df

baris = []
t0 = time.time()
for sk, th in LUAR:
    latih_luar = df[df.tahun < th] if sk == "S1" else df[df.tahun != th]
    for v in sorted(latih_luar.tahun.unique()):
        latih, uji = latih_luar[latih_luar.tahun != v], latih_luar[latih_luar.tahun == v]
        skor = {"klimatologi": R.klim_rata(latih, uji), **R.baseline_baru(latih, uji, YLAB)}
        skor["xgb"] = R.dipelajari_xgb(latih, uji, YLAB)
        b = R.bingkai(uji, YLAB, sk, str(th), skor)
        b["tahun_dalam"] = v
        baris.append(b)
    print(f"{sk} {th}: {latih_luar.tahun.nunique()} tahun dalam, {time.time() - t0:.0f} dtk", flush=True)
hasil = pd.concat(baris, ignore_index=True)
hasil.to_csv(OUT, index=False)
print(f"tersimpan: {OUT}  baris {len(hasil)}")
