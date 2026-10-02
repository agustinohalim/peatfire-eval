"""Kelompok fitur panel nasional Artikel 2; dipakai patokan_prediksi.py dan patokan_ablasi.py.

Urutan dalam tiap kelompok dipertahankan supaya putaran lama (dasar, hujan) memakai urutan
kolom yang sama dengan sebelum berkas ini ada.

Putaran (PATOKAN_PUTARAN, bawaan "penuh"):
  lengkap : enam kelompok, panel_nasional_fitur_lengkap.csv — putaran naskah s.d. v2.0.1
  penuh   : + gambut (gabung_gambut_nasional.py) + cuaca kebakaran FWI (gabung_fwi_nasional.py),
            panel_nasional_fitur_penuh.csv; baseline FWI ikut dihitung. Putaran naskah sejak v2.1.0.
Semua skrip analisis membaca nama berkas dari sini, jadi satu variabel lingkungan memilih putaran.
"""

import os

KELOMPOK = {
    "api": ["tp_lag1", "tp_lag2", "tp_lag3", "tp_lag12"],
    "musim": ["bulan_sin", "bulan_cos"],
    "iklim": [f"oni_lag{l}" for l in range(1, 7)] + [f"dmi_lag{l}" for l in range(1, 4)],
    "hujan": [f"hujan_lag{l}" for l in (1, 2, 3)] + [f"hujan_anom_lag{l}" for l in (1, 2, 3)],
    "lahan": ["elevasi_m", "lereng_deg", "tutupan_pohon_2000", "lc_hutan", "lc_semak_savana",
              "lc_lahan_basah", "lc_pertanian", "hilang_hutan_lalu"],
    "manusia": ["log_kepadatan", "lc_perkotaan"],
}
TAMBAHAN_PENUH = {
    "gambut": ["gambut_fraksi"],
    "cuaca": ["fwi_lag1", "fwi_lag2", "fwi_lag3", "fwi_maks_lag1", "fwi_anom_lag1"],
}

PUTARAN = os.environ.get("PATOKAN_PUTARAN", "penuh")
assert PUTARAN in ("lengkap", "penuh"), PUTARAN
if PUTARAN == "penuh":
    KELOMPOK = {**KELOMPOK, **TAMBAHAN_PENUH}
PANEL = f"panel_nasional_fitur_{PUTARAN}.csv"
PREDIKSI = f"patokan_prediksi_{PUTARAN}.csv"
ABLASI = "patokan_ablasi.csv" if PUTARAN == "lengkap" else "patokan_ablasi_penuh.csv"

# Model prakiraan yang dibandingkan di tabel, provinsi, dan uji panel. fwi_kini (FWI bulan
# berjalan) BUKAN prakiraan — cuaca serentak — dan hanya dilaporkan sebagai rujukan terpisah.
MODEL = ["klimatologi", "persistence", "seasonal_naive", "rasio", "logistik", "rf", "xgb", "mlp"]
if PUTARAN == "penuh":
    MODEL = MODEL[:4] + ["fwi", "fwi_logistik"] + MODEL[4:]
