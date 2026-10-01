"""
Artikel 2, Percobaan C — ablasi kelompok fitur dengan gradient boosting.

Panel: DL_FIRE_NASIONAL/panel_nasional_fitur_lengkap.csv (gabung_lahan_nasional.py).
Kelompok: kelompok_fitur.KELOMPOK (api, musim, iklim, hujan, lahan, manusia).

Konfigurasi: "lengkap" (semua kelompok) dan "tanpa_<k>" untuk tiap kelompok k. Hanya gradient
boosting (parameter Artikel 1), karena ablasi bertanya tentang fitur, bukan model; dan hanya
skema S1 (2019-2025) dan S2 (2015, 2014, 2019), dua sasaran. Transfer antarpulau tidak diulang
di sini.

Keluaran: DL_FIRE_NASIONAL/patokan_ablasi.csv — satu kolom skor per konfigurasi, baris sama
dengan patokan_prediksi*.csv (gid, tahun, bulan, sasaran, skema, tag), plus klimatologi untuk
pembanding dalam-tahun.

Pakai:  python patokan_ablasi.py            (latar belakang; ± 20-40 menit)
"""

import os
import warnings
import pandas as pd
from xgboost import XGBClassifier
from kelompok_fitur import KELOMPOK

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
df = pd.read_csv(os.path.join(D, "panel_nasional_fitur_lengkap.csv"))
SEMUA = [c for k in KELOMPOK for c in KELOMPOK[k]]
KONFIG = {"lengkap": SEMUA}
KONFIG.update({f"tanpa_{k}": [c for c in SEMUA if c not in KELOMPOK[k]] for k in KELOMPOK})


def xgb(latih, uji, ylab, kolom):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                      random_state=42, verbosity=0, n_jobs=-1)
    m.fit(latih[kolom].fillna(0.0), latih[ylab])
    return m.predict_proba(uji[kolom].fillna(0.0))[:, 1]


def klim(latih, uji):
    k = latih.groupby(["gid", "bulan_ke"])["titik_panas"].mean()
    return k.reindex(pd.MultiIndex.from_arrays([uji["gid"], uji["bulan_ke"]])).fillna(0.0).to_numpy()


baris = []
for ylab in ("y_kabupaten", "y_gabungan"):
    lipatan = [("S1", str(t), df[df["tahun"] < t], df[df["tahun"] == t]) for t in range(2019, 2026)]
    lipatan += [("S2", str(t), df[df["tahun"] != t], df[df["tahun"] == t]) for t in (2015, 2014, 2019)]
    for skema, tag, latih, uji in lipatan:
        b = uji[["gid", "provinsi", "tahun", "bulan", "ada_positif"]].copy()
        b["y"], b["sasaran"], b["skema"], b["tag"] = uji[ylab].to_numpy(), ylab, skema, tag
        b["klimatologi"] = klim(latih, uji)
        for nama, kolom in KONFIG.items():
            b[nama] = xgb(latih, uji, ylab, kolom)
        baris.append(b)
        print(f"{ylab} {skema} {tag}", flush=True)

out = pd.concat(baris, ignore_index=True)
out.to_csv(os.path.join(D, "patokan_ablasi.csv"), index=False)
print(f"tersimpan: patokan_ablasi.csv  baris: {len(out)}  konfigurasi: {', '.join(KONFIG)}")
