"""
Menggabungkan curah hujan bulanan CHIRPS per kabupaten (ekspor Skrip_CHIRPS_Earth_Engine.js,
25 September 2026) ke panel fitur nasional Artikel 2.

Masukan : DL_FIRE_NASIONAL/chirps_bulanan_kabupaten_2012_2025.csv  (gid, bulan, hujan_mm)
          DL_FIRE_NASIONAL/panel_nasional_fitur_bersih.csv
Keluaran: DL_FIRE_NASIONAL/panel_nasional_fitur_hujan.csv  (berkas bersih tidak diubah)

Fitur baru, semuanya jeda (hujan bulan berjalan TIDAK dipakai: hujan dan kebakaran serentak,
jadi itu kebocoran):
  hujan_lag1..3       : curah hujan bulan t-1..t-3, mm
  hujan_anom_lag1..3  : selisih terhadap rata-rata kabupaten-bulan kalender 2012-2018, mm.
                        Garis dasar berhenti 2018 supaya tidak memakai tahun uji kronologis
                        (2019-2025). Pada skema tahun ekstrem disisihkan (2014, 2015) garis
                        dasar memuat tahun uji; yang bocor hanya rata-rata hujan, bukan sasaran.
                        Dicatat di Hasil_Patokan_3.

Satu satuan tanpa nilai: Kepulauan Seribu (IDN.7.6_1), gugus pulau yang lebih kecil dari satu
sel CHIRPS 0,05 derajat, null di semua 168 bulan. Diisi dari Jakarta Utara (IDN.7.5_1), daratan
terdekat, dan ditandai kolom hujan_imputasi = 1. Tidak diisi nol: nol berarti tidak hujan.

Pakai:  python gabung_chirps_nasional.py
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
KOSONG, TETANGGA = "IDN.7.6_1", "IDN.7.5_1"

c = pd.read_csv(os.path.join(D, "chirps_bulanan_kabupaten_2012_2025.csv"))
assert len(c) == 502 * 168 and not c.duplicated(["gid", "bulan"]).any()
null_gid = set(c.loc[c["hujan_mm"].isna(), "gid"])
assert null_gid == {KOSONG}, null_gid
isi = c[c["gid"] == TETANGGA].set_index("bulan")["hujan_mm"]
c["hujan_imputasi"] = (c["gid"] == KOSONG).astype(int)
c.loc[c["gid"] == KOSONG, "hujan_mm"] = c.loc[c["gid"] == KOSONG, "bulan"].map(isi).to_numpy()
assert c["hujan_mm"].notna().all()

c["tahun"] = c["bulan"].str[:4].astype(int)
c["bulan_ke"] = c["bulan"].str[5:7].astype(int)
c = c.sort_values(["gid", "bulan"])
normal = c[c["tahun"] <= 2018].groupby(["gid", "bulan_ke"])["hujan_mm"].mean().rename("normal")
c = c.join(normal, on=["gid", "bulan_ke"])
g = c.groupby("gid")
for l in (1, 2, 3):
    c[f"hujan_lag{l}"] = g["hujan_mm"].shift(l)
    c[f"hujan_anom_lag{l}"] = c[f"hujan_lag{l}"] - g["normal"].shift(l)

f = pd.read_csv(os.path.join(D, "panel_nasional_fitur_bersih.csv"))
kol = ["gid", "bulan", "hujan_imputasi"] + [f"hujan_lag{l}" for l in (1, 2, 3)] \
    + [f"hujan_anom_lag{l}" for l in (1, 2, 3)]
out = f.merge(c[kol], on=["gid", "bulan"], how="left", validate="one_to_one")
assert len(out) == len(f) and out[kol[3:]].notna().all().all(), "ada fitur hujan kosong"
out.to_csv(os.path.join(D, "panel_nasional_fitur_hujan.csv"), index=False)
print(f"{len(out)} baris, {out['gid'].nunique()} satuan; kolom baru: {', '.join(kol[2:])}")
print(f"korelasi Spearman hujan_lag1 vs titik_panas: "
      f"{out[['hujan_lag1', 'titik_panas']].corr('spearman').iloc[0, 1]:.3f}")
