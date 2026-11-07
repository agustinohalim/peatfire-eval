"""
Fase G3 (Hasil_Patokan_12): kelas keyakinan MODIS untuk bangun_panel_nasional.js.

Keyakinan MODIS (Collection 6/6.1) berupa angka 0-100; kelasnya rendah < 30, nominal 30-79, tinggi >= 80.
Skrip ini menulis ulang tiap CSV DL_FIRE_MODIS/modis_sp_*.csv ke DL_FIRE_MODIS/kelas/ dengan kolom
`confidence` diganti `l` / `n` / `h`, kolom lain apa adanya, supaya bangun_panel_nasional.js (saringan
`confidence != l`, `type == 0`) dapat dipakai tanpa perubahan.

Pakai:
    python modis_keyakinan.py
    node bangun_panel_nasional.js DL_FIRE_SV-C2_792597/gadm41_IDN_2.json DL_FIRE_MODIS/panel_modis.csv DL_FIRE_MODIS/kelas
"""
import csv
import glob
import os
from collections import Counter

ASAL = "DL_FIRE_MODIS"
TUJUAN = os.path.join(ASAL, "kelas")
os.makedirs(TUJUAN, exist_ok=True)
hitung, versi = Counter(), Counter()
berkas = sorted(glob.glob(os.path.join(ASAL, "modis_sp_*.csv")))
for f in berkas:
    with open(f, encoding="utf-8", newline="") as a, open(os.path.join(TUJUAN, os.path.basename(f)), "w",
                                                         encoding="utf-8", newline="") as b:
        r = csv.DictReader(a)
        w = csv.DictWriter(b, fieldnames=r.fieldnames, lineterminator="\n")
        w.writeheader()
        for baris in r:
            c = int(float(baris["confidence"]))
            k = "l" if c < 30 else ("n" if c < 80 else "h")
            baris["confidence"] = k
            hitung[k] += 1
            versi[baris.get("version", "")] += 1
            w.writerow(baris)
print(f"berkas {len(berkas)}; kelas {dict(hitung)}; versi {dict(versi)}")
