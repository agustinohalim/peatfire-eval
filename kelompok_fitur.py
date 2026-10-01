"""Kelompok fitur panel nasional Artikel 2; dipakai patokan_prediksi.py dan patokan_ablasi.py.

Urutan dalam tiap kelompok dipertahankan supaya putaran lama (dasar, hujan) memakai urutan
kolom yang sama dengan sebelum berkas ini ada.
"""

KELOMPOK = {
    "api": ["tp_lag1", "tp_lag2", "tp_lag3", "tp_lag12"],
    "musim": ["bulan_sin", "bulan_cos"],
    "iklim": [f"oni_lag{l}" for l in range(1, 7)] + [f"dmi_lag{l}" for l in range(1, 4)],
    "hujan": [f"hujan_lag{l}" for l in (1, 2, 3)] + [f"hujan_anom_lag{l}" for l in (1, 2, 3)],
    "lahan": ["elevasi_m", "lereng_deg", "tutupan_pohon_2000", "lc_hutan", "lc_semak_savana",
              "lc_lahan_basah", "lc_pertanian", "hilang_hutan_lalu"],
    "manusia": ["log_kepadatan", "lc_perkotaan"],
}
