"""
Membersihkan panel nasional untuk Artikel 2 sesuai Kartu_Data_Nasional.md bagian 3 dan 4.
Berkas asli TIDAK diubah; keluaran ke berkas baru berakhiran _bersih.

  1. Empat badan air (tipe WaterBody: Waduk Cirata, Waduk Kedungombo, Danau, Lake Toba)
     dikeluarkan: 502 -> 498 satuan.
  2. Lima kabupaten Kalimantan Utara tanpa tipe di GADM diisi: Tarakan = Kota, lainnya Kabupaten.
  3. Dua sasaran ditambahkan, dihitung pada 2012-2025:
       y_kabupaten : titik_panas > maks(persentil 90 kabupaten itu, 10)   <- sasaran utama
       y_gabungan  : titik_panas > persentil 90 seluruh satuan (gabungan)  <- pembanding
     Batas bawah 10 diputuskan 25 September 2026 (Kartu_Data_Nasional.md bagian 3).
     Kolom ada_positif menandai kabupaten dengan sedikitnya satu bulan y_kabupaten; kabupaten
     tanpa positif tetap di panel tetapi dikeluarkan dari metrik per kabupaten.

Pemetaan provinsi hasil pemekaran Papua 2022 BELUM dibuat (Kartu bagian 4.3): perlu dicocokkan
dengan undang-undang pembentukannya sebelum split transfer antarprovinsi memakainya.

Pakai:
    python bersihkan_panel_nasional.py

Keluaran: DL_FIRE_NASIONAL/panel_nasional_bersih.csv dan panel_nasional_fitur_bersih.csv
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
BATAS_BAWAH = 10
TIPE_KALTARA = {"Tarakan": "Kota", "Bulungan": "Kabupaten", "Malinau": "Kabupaten",
                "Nunukan": "Kabupaten", "TanaTidung": "Kabupaten"}


def bersihkan(df):
    n0 = df["gid"].nunique()
    air = df.loc[df["tipe"] == "WaterBody", "kabupaten"].unique().tolist()
    df = df[df["tipe"] != "WaterBody"].copy()
    kosong = df["tipe"].isna()
    df.loc[kosong, "tipe"] = df.loc[kosong, "kabupaten"].map(TIPE_KALTARA)
    assert df["tipe"].notna().all(), "masih ada satuan tanpa tipe"
    df = df[df["tahun"] <= 2025]
    q_kab = df.groupby("gid")["titik_panas"].quantile(0.90)
    ambang_kab = np.maximum(df["gid"].map(q_kab), BATAS_BAWAH)
    ambang_gab = df["titik_panas"].quantile(0.90)
    df["ambang_kabupaten"] = ambang_kab
    df["y_kabupaten"] = (df["titik_panas"] > ambang_kab).astype(int)
    df["y_gabungan"] = (df["titik_panas"] > ambang_gab).astype(int)
    ada = df.groupby("gid")["y_kabupaten"].max()
    df["ada_positif"] = df["gid"].map(ada).astype(int)
    return df, n0, air, ambang_gab


# Ambang dihitung SEKALI dari panel lengkap 2012-2025, lalu diterapkan ke berkas fitur. Berkas
# fitur mulai 2013 (lag 12 bulan), jadi menghitung ambang di sana akan memberi angka lain.
panel = pd.read_csv(os.path.join(D, "panel_nasional.csv"))
panel, n0, air, ambang_gab = bersihkan(panel)
ambang_kab = panel.groupby("gid")["ambang_kabupaten"].first()
ada = panel.groupby("gid")["ada_positif"].first()
panel.to_csv(os.path.join(D, "panel_nasional_bersih.csv"), index=False)

fitur = pd.read_csv(os.path.join(D, "panel_nasional_fitur.csv"))
fitur = fitur[fitur["tipe"] != "WaterBody"].copy()
kosong = fitur["tipe"].isna()
fitur.loc[kosong, "tipe"] = fitur.loc[kosong, "kabupaten"].map(TIPE_KALTARA)
fitur = fitur[fitur["tahun"] <= 2025]
fitur["ambang_kabupaten"] = fitur["gid"].map(ambang_kab)
fitur["y_kabupaten"] = (fitur["titik_panas"] > fitur["ambang_kabupaten"]).astype(int)
fitur["y_gabungan"] = (fitur["titik_panas"] > ambang_gab).astype(int)
fitur["ada_positif"] = fitur["gid"].map(ada).astype(int)
assert fitur["ambang_kabupaten"].notna().all() and fitur["tipe"].notna().all()
fitur.to_csv(os.path.join(D, "panel_nasional_fitur_bersih.csv"), index=False)

print(f"satuan: {n0} -> {panel['gid'].nunique()} (dikeluarkan: {', '.join(air)})")
print(f"ambang gabungan > {ambang_gab:.0f} (dari panel 2012-2025), batas bawah per kabupaten {BATAS_BAWAH}")
print(f"panel  : {len(panel)} baris; prevalensi y_kabupaten {panel['y_kabupaten'].mean() * 100:.2f}%, "
      f"y_gabungan {panel['y_gabungan'].mean() * 100:.2f}%; kabupaten tanpa positif "
      f"{(ada == 0).sum()}")
print(f"fitur  : {len(fitur)} baris (2013-2025); prevalensi y_kabupaten {fitur['y_kabupaten'].mean() * 100:.2f}%, "
      f"y_gabungan {fitur['y_gabungan'].mean() * 100:.2f}%")
