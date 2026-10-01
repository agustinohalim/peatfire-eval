"""
Menggabungkan lapisan lahan dan manusia (Skrip_Statis_Earth_Engine.js, 25 September 2026) ke
panel fitur nasional yang sudah memuat hujan.

Masukan : DL_FIRE_NASIONAL/panel_nasional_fitur_hujan.csv
          DL_FIRE_NASIONAL/statis_kabupaten.csv          (GID_2, luas_km2, elevasi_m, lereng_deg,
                                                          tutupan_pohon_2000, penduduk_2020)
          DL_FIRE_NASIONAL/tutupan_tahunan.csv           (GID_2, tahun 2011-2023, lima fraksi)
          DL_FIRE_NASIONAL/hilang_hutan_tahunan_v2.csv   (GID_2, tahun 2011-2025, hilang_hutan)
Keluaran: DL_FIRE_NASIONAL/panel_nasional_fitur_lengkap.csv

Fitur baru dan kelompoknya (untuk ablasi Percobaan C):
  lahan   : elevasi_m, lereng_deg, tutupan_pohon_2000, lc_hutan, lc_semak_savana,
            lc_lahan_basah, lc_pertanian, hilang_hutan_lalu
  manusia : log_kepadatan (log1p penduduk 2020 per km2), lc_perkotaan

Jeda satu tahun untuk semua lapisan tahunan: bulan di tahun Y memakai tutupan dan kehilangan
hutan tahun Y-1, karena kehilangan hutan pada tahun kebakaran sebagian AKIBAT kebakaran itu.
MCD12Q1 berhenti 2023, jadi bulan tahun 2025 memakai tutupan 2023 (jeda dua tahun) — dicatat,
tidak diisi dengan cara lain. Penduduk hanya satu tahun (2020): peubah ruang, bukan waktu.
WorldPop cenderung melebihi sensus di perdesaan (Pulang Pisau 157 ribu lawan sekitar 133 ribu
sensus 2020); dipakai sebagai kepadatan relatif antarkabupaten.

Pakai:  python gabung_lahan_nasional.py
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
LAHAN = ["elevasi_m", "lereng_deg", "tutupan_pohon_2000", "lc_hutan", "lc_semak_savana",
         "lc_lahan_basah", "lc_pertanian", "hilang_hutan_lalu"]
MANUSIA = ["log_kepadatan", "lc_perkotaan"]

f = pd.read_csv(os.path.join(D, "panel_nasional_fitur_hujan.csv"))
s = pd.read_csv(os.path.join(D, "statis_kabupaten.csv")).rename(columns={"GID_2": "gid"})
t = pd.read_csv(os.path.join(D, "tutupan_tahunan.csv")).rename(columns={"GID_2": "gid"})
h = pd.read_csv(os.path.join(D, "hilang_hutan_tahunan_v2.csv")).rename(columns={"GID_2": "gid"})
for d, n in ((s, 502), (t, 502 * 13), (h, 502 * 15)):
    assert len(d) == n and d.notna().all().all(), (len(d), n)
t["tahun"], h["tahun"] = t["tahun"].astype(int), h["tahun"].astype(int)

# Pemeriksaan kewajaran versi kedua: jumlah per kabupaten 2011-2025 tidak boleh mendekati 1 di
# semua kabupaten (gejala versi pertama yang salah), dan tidak boleh melebihi 1.
tot = h.groupby("gid")["hilang_hutan"].sum()
assert tot.max() <= 1 and tot.median() < 0.5, tot.describe()
nas = (h.merge(s[["gid", "luas_km2"]], on="gid").assign(km2=lambda x: x.hilang_hutan * x.luas_km2)
       .groupby("tahun")["km2"].sum())
print("kehilangan tutupan pohon nasional (502 satuan), km2:", nas.round(0).astype(int).to_dict())

s["log_kepadatan"] = np.log1p(s["penduduk_2020"] / s["luas_km2"])
t = t.rename(columns={c: f"lc_{c}" for c in ("hutan", "semak_savana", "lahan_basah", "pertanian", "perkotaan")})
t["tahun"] = t["tahun"] + 1                       # nilai tahun Y-1 dipakai di tahun Y
h = h.rename(columns={"hilang_hutan": "hilang_hutan_lalu"})
h["tahun"] = h["tahun"] + 1

out = f.merge(s[["gid", "elevasi_m", "lereng_deg", "tutupan_pohon_2000", "log_kepadatan"]],
              on="gid", how="left", validate="many_to_one")
# Tutupan: tahun 2025 belum ada (MCD12Q1 berhenti 2023 -> tersedia sampai tahun fitur 2024).
lc = [c for c in t.columns if c.startswith("lc_")]
t_isi = pd.concat([t, t[t["tahun"] == 2024].assign(tahun=2025)], ignore_index=True)
out = out.merge(t_isi, on=["gid", "tahun"], how="left", validate="many_to_one")
out = out.merge(h, on=["gid", "tahun"], how="left", validate="many_to_one")
assert len(out) == len(f) and out[LAHAN + MANUSIA].notna().all().all(), \
    out[LAHAN + MANUSIA].isna().sum()
out.to_csv(os.path.join(D, "panel_nasional_fitur_lengkap.csv"), index=False)
print(f"{len(out)} baris, {out['gid'].nunique()} satuan; lahan: {', '.join(LAHAN)}; manusia: {', '.join(MANUSIA)}")
r = out[LAHAN + MANUSIA + ["titik_panas"]].corr("spearman")["titik_panas"].drop("titik_panas")
print("korelasi Spearman dengan titik_panas:", r.round(3).to_dict())
