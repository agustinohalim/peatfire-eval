#!/usr/bin/env python3
"""Unduh MODIS (Terra dan Aqua, Collection 6.1 standard processing) seluruh Indonesia 2003-2025 lewat API FIRMS.

Fase G revisi PeatFireBench (`Hasil_Patokan_12.md`, G3): uji EF-L pada tahun 2003-2011 yang tidak pernah
dinilai. Salinan `unduh_firms_nasional.py`; yang berubah hanya sumber (`MODIS_SP`), rentang (sejak
1 Januari 2003, Aqua lengkap), direktori keluaran dan awalan nama berkas. Sifat dapat-dilanjutkan, penulisan
lewat berkas sementara dan pencatatan kegagalan sama persis.

Pemakaian:
    $env:FIRMS_MAP_KEY = '...'            # PowerShell; kunci tidak pernah ditulis ke berkas
    python unduh_firms_modis.py
    python unduh_firms_modis.py --uji     # tiga jendela saja

Keluaran: DL_FIRE_MODIS/modis_sp_YYYY-MM-DD.csv (dikecualikan .gitignore lewat DL_FIRE_*/ dan *.csv).
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASIS = "https://firms.modaps.eosdis.nasa.gov/api"
SUMBER = "MODIS_SP"

# Kotak unduh saja. Penyaringan analisis tetap poligon kabupaten GADM,
# `confidence != l`, dan `type == 0`. Memakai kotak sebagai batas wilayah
# adalah kekeliruan yang justru menjadi temuan Artikel 1 bagian 3.3.
KOTAK = "95,-11,141,6"

HARI_PER_PERMINTAAN = 5  # batas server: Invalid day range. Expects [1..5]

# Aqua lengkap sejak 2003; 2026 tidak diambil (sama dengan panel VIIRS).
AWAL_ARSIP = date(2003, 1, 1)
AKHIR_ARSIP = date(2025, 12, 31)

KELUARAN = Path("DL_FIRE_MODIS")
JEDA_DETIK = 2
PERCOBAAN_ULANG = 3


def jendela(mulai, selesai, lebar):
    """Hasilkan tanggal mulai tiap jendela."""
    kini = mulai
    while kini <= selesai:
        yield kini
        kini += timedelta(days=lebar)


def unduh_satu(kunci, tanggal, hari):
    """Kembalikan (berhasil, teks_atau_pesan). Mencoba ulang bila sementara."""
    url = f"{BASIS}/area/csv/{kunci}/{SUMBER}/{KOTAK}/{hari}/{tanggal.isoformat()}"
    permintaan = urllib.request.Request(
        url, headers={"User-Agent": "peatfire-eval/unduh-modis (riset akademik)"}
    )
    for percobaan in range(1, PERCOBAAN_ULANG + 1):
        try:
            with urllib.request.urlopen(permintaan, timeout=120) as tanggapan:
                teks = tanggapan.read().decode("utf-8", "replace")
            kepala = teks.splitlines()[0].lower() if teks.strip() else ""
            if "latitude" not in kepala:
                return False, f"bukan CSV: {teks[:120].strip()}"
            return True, teks
        except urllib.error.HTTPError as galat:
            # 4xx tidak akan membaik dengan mencoba ulang, 5xx mungkin.
            if galat.code < 500:
                return False, f"HTTP {galat.code}"
            pesan = f"HTTP {galat.code}"
        except Exception as galat:  # noqa: BLE001
            pesan = f"{type(galat).__name__}: {galat}"
        if percobaan < PERCOBAAN_ULANG:
            time.sleep(JEDA_DETIK * 2 * percobaan)
    return False, pesan


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mulai", type=date.fromisoformat, default=AWAL_ARSIP)
    p.add_argument("--selesai", type=date.fromisoformat, default=AKHIR_ARSIP)
    p.add_argument("--uji", action="store_true", help="tiga jendela saja, untuk mencoba")
    argumen = p.parse_args()

    kunci = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not kunci:
        print("FIRMS_MAP_KEY tidak ada di environment.")
        print("    $env:FIRMS_MAP_KEY = '...'      # PowerShell")
        print("    export FIRMS_MAP_KEY=...        # bash")
        return 1

    if argumen.mulai < AWAL_ARSIP:
        print(f"Arsip dipakai mulai {AWAL_ARSIP}. Tanggal mulai disesuaikan.")
        argumen.mulai = AWAL_ARSIP

    semua = list(jendela(argumen.mulai, argumen.selesai, HARI_PER_PERMINTAAN))
    if argumen.uji:
        semua = semua[:3]

    KELUARAN.mkdir(exist_ok=True)

    print(f"Sumber       : {SUMBER}")
    print(f"Kotak unduh  : {KOTAK}")
    print(f"Rentang      : {argumen.mulai} sampai {argumen.selesai}")
    print(f"Jendela      : {len(semua)} × {HARI_PER_PERMINTAAN} hari")
    print(f"Keluaran     : {KELUARAN}/")
    print(f"Taksiran waktu: ~{len(semua) * JEDA_DETIK / 60:.0f} menit")
    print()

    baru = dilewati = 0
    baris_total = 0
    gagal = []
    mulai_jam = time.time()

    for i, tanggal in enumerate(semua, 1):
        berkas = KELUARAN / f"modis_sp_{tanggal.isoformat()}.csv"
        if berkas.exists():
            dilewati += 1
            continue

        berhasil, muatan = unduh_satu(kunci, tanggal, HARI_PER_PERMINTAAN)
        if not berhasil:
            gagal.append((tanggal.isoformat(), muatan))
            print(f"[{i:4d}/{len(semua)}] {tanggal}  GAGAL  {muatan}")
            continue

        # Tulis lewat berkas sementara supaya tidak pernah ada CSV separuh jadi.
        sementara = berkas.with_suffix(".csv.parsial")
        sementara.write_text(muatan, encoding="utf-8")
        sementara.replace(berkas)

        n = max(0, len(muatan.strip().splitlines()) - 1)
        baris_total += n
        baru += 1

        if baru % 20 == 0 or i == len(semua):
            lewat = time.time() - mulai_jam
            sisa = (len(semua) - i) * JEDA_DETIK / 60
            print(
                f"[{i:4d}/{len(semua)}] {tanggal}  {n:>6,} baris  "
                f"total {baris_total:>9,}  sisa ~{sisa:.0f} mnt"
            )

        time.sleep(JEDA_DETIK)

    print()
    print("=" * 70)
    print(f"Berkas baru   : {baru:,}")
    print(f"Sudah ada     : {dilewati:,}")
    print(f"Gagal         : {len(gagal):,}")
    print(f"Baris terunduh: {baris_total:,}")
    print(f"Waktu         : {(time.time() - mulai_jam) / 60:.1f} menit")

    if gagal:
        print()
        print("Jendela yang gagal — jalankan ulang skrip untuk mengambilnya,")
        print("yang sudah ada akan dilewati:")
        for tanggal, sebab in gagal[:20]:
            print(f"  {tanggal}  {sebab}")
        if len(gagal) > 20:
            print(f"  ... dan {len(gagal) - 20} lagi")
        return 2

    print()
    print("Selesai tanpa kegagalan.")
    print()
    print("Berikutnya: modis_keyakinan.py, lalu bangun_panel_nasional.js (Hasil_Patokan_12, G3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
