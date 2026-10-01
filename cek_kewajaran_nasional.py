#!/usr/bin/env python3
"""Periksa kewajaran unduhan FIRMS nasional sebelum panel dibangun.

Langkah 1 dari tiga yang disebut `unduh_firms_nasional.py` di akhir jalannya,
dan bagian I `Artikel_2_Rencana.md`: periksa kewajaran per tahun sebelum apa
pun. Menemukan data cacat sekarang jauh lebih murah daripada menemukannya
setelah panel, model, dan gambar dibangun di atasnya.

Yang diperiksa:

  1. Kelengkapan berkas — apakah ada jendela lima hari yang bolong
  2. Keutuhan CSV — header seragam, jumlah kolom konsisten
  3. Sebaran per tahun, sebelum dan sesudah penyaringan analisis
  4. Sebaran per bulan — puncaknya harus Agustus-Oktober
  5. Rentang koordinat — harus di dalam kotak unduh
  6. Nilai `confidence` dan `type` yang benar-benar muncul

Penyaringan analisis yang dipakai di sini sama persis dengan Artikel 1 dan
dengan aturan pada README: `confidence != l` dan `type == 0`. Penyaringan
poligon kabupaten TIDAK dikerjakan di sini — itu langkah pembangunan panel,
dan perlu GADM.

Pemakaian:
    python cek_kewajaran_nasional.py
"""

import csv
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

SUMBER_DIR = Path("DL_FIRE_NASIONAL")
AWAL = date(2012, 1, 20)
AKHIR = date(2025, 12, 31)
LEBAR = 5

# Kotak unduh, untuk memeriksa tidak ada titik di luarnya.
LON_MIN, LON_MAX = 95.0, 141.0
LAT_MIN, LAT_MAX = -11.0, 6.0


def judul(teks):
    print()
    print("=" * 74)
    print(teks)
    print("=" * 74)


def main():
    if not SUMBER_DIR.is_dir():
        print(f"{SUMBER_DIR}/ tidak ada. Jalankan unduh_firms_nasional.py dahulu.")
        return 1

    # ------------------------------------------------------ 1. kelengkapan
    judul("1. Kelengkapan berkas")
    diharapkan = []
    kini = AWAL
    while kini <= AKHIR:
        diharapkan.append(kini)
        kini += timedelta(days=LEBAR)
    ada = {p.stem.replace("viirs_snpp_sp_", "") for p in SUMBER_DIR.glob("*.csv")}
    bolong = [d.isoformat() for d in diharapkan if d.isoformat() not in ada]
    print(f"Jendela diharapkan : {len(diharapkan):,}")
    print(f"Berkas ada         : {len(ada):,}")
    if bolong:
        print(f"BOLONG             : {len(bolong):,}")
        for b in bolong[:15]:
            print(f"  {b}")
        if len(bolong) > 15:
            print(f"  ... dan {len(bolong) - 15} lagi")
        print()
        print("Jalankan ulang unduh_firms_nasional.py — yang sudah ada dilewati.")
    else:
        print("BOLONG             : tidak ada")

    # -------------------------------- 2-6. baca sekali, kumpulkan semuanya
    judul("2. Keutuhan CSV dan pembacaan")
    per_tahun_mentah = Counter()
    per_tahun_saring = Counter()
    per_bulan_saring = Counter()
    confidence = Counter()
    tipe = Counter()
    header_beda = []
    baris_cacat = 0
    lon_lo = lat_lo = 999.0
    lon_hi = lat_hi = -999.0
    luar_kotak = 0
    header_acuan = None

    for berkas in sorted(SUMBER_DIR.glob("*.csv")):
        with berkas.open(encoding="utf-8", newline="") as f:
            pembaca = csv.DictReader(f)
            if header_acuan is None:
                header_acuan = pembaca.fieldnames
            elif pembaca.fieldnames != header_acuan:
                header_beda.append(berkas.name)
                continue
            for baris in pembaca:
                try:
                    tanggal = baris["acq_date"]
                    tahun = int(tanggal[:4])
                    bulan = int(tanggal[5:7])
                    lon = float(baris["longitude"])
                    lat = float(baris["latitude"])
                except (KeyError, ValueError, TypeError):
                    baris_cacat += 1
                    continue

                per_tahun_mentah[tahun] += 1
                confidence[baris.get("confidence", "?")] += 1
                tipe[baris.get("type", "?")] += 1

                lon_lo, lon_hi = min(lon_lo, lon), max(lon_hi, lon)
                lat_lo, lat_hi = min(lat_lo, lat), max(lat_hi, lat)
                if not (LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX):
                    luar_kotak += 1

                # Penyaringan analisis, sama persis dengan Artikel 1.
                if baris.get("confidence") != "l" and baris.get("type") == "0":
                    per_tahun_saring[tahun] += 1
                    per_bulan_saring[bulan] += 1

    print(f"Header acuan  : {', '.join(header_acuan or [])}")
    print(f"Header berbeda: {len(header_beda)}")
    for n in header_beda[:5]:
        print(f"  {n}")
    print(f"Baris cacat   : {baris_cacat:,}")

    # ------------------------------------------------------ 3. per tahun
    judul("3. Sebaran per tahun")
    total_mentah = sum(per_tahun_mentah.values())
    total_saring = sum(per_tahun_saring.values())
    print(f"{'Tahun':<7}{'Mentah':>12}{'Tersaring':>12}{'Lolos':>8}")
    for tahun in sorted(per_tahun_mentah):
        m = per_tahun_mentah[tahun]
        s = per_tahun_saring[tahun]
        print(f"{tahun:<7}{m:>12,}{s:>12,}{s / m * 100 if m else 0:>7.1f}%")
    print(f"{'TOTAL':<7}{total_mentah:>12,}{total_saring:>12,}"
          f"{total_saring / total_mentah * 100 if total_mentah else 0:>7.1f}%")

    if per_tahun_saring:
        urut = sorted(per_tahun_saring.items(), key=lambda x: -x[1])
        print()
        print("Tiga tahun terparah setelah penyaringan:")
        for tahun, n in urut[:3]:
            print(f"  {tahun}  {n:,}")
        print("Tiga tahun paling sepi:")
        for tahun, n in urut[-3:]:
            print(f"  {tahun}  {n:,}")
        print()
        print("PERIKSA SENDIRI: 2015 dan 2019 harus muncul di jajaran teratas.")
        print("Kalau tidak, ada yang salah pada unduhan atau penyaringan.")

    # ------------------------------------------------------ 4. per bulan
    judul("4. Sebaran per bulan, setelah penyaringan")
    nama = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
            "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    puncak = max(per_bulan_saring.values()) if per_bulan_saring else 1
    for b in range(1, 13):
        n = per_bulan_saring.get(b, 0)
        bar = "#" * int(40 * n / puncak) if puncak else ""
        print(f"  {nama[b - 1]}  {n:>10,}  {bar}")
    if per_bulan_saring:
        tertinggi = max(per_bulan_saring, key=per_bulan_saring.get)
        print()
        print(f"Bulan tertinggi: {nama[tertinggi - 1]}")
        print("PERIKSA SENDIRI: puncaknya harus Agustus, September, atau Oktober.")

    # -------------------------------------------- 5-6. koordinat dan nilai
    judul("5. Rentang koordinat")
    print(f"Longitude: {lon_lo:.4f} sampai {lon_hi:.4f}   (kotak {LON_MIN}-{LON_MAX})")
    print(f"Latitude : {lat_lo:.4f} sampai {lat_hi:.4f}   (kotak {LAT_MIN}-{LAT_MAX})")
    print(f"Di luar kotak: {luar_kotak:,}")

    judul("6. Nilai confidence dan type")
    for nilai, n in confidence.most_common():
        print(f"  confidence {nilai!r:<6} {n:>12,}")
    for nilai, n in tipe.most_common():
        print(f"  type       {nilai!r:<6} {n:>12,}")
    print()
    print("type 0 = vegetation fire. Selain itu dibuang penyaringan analisis.")

    judul("Kesimpulan")
    masalah = []
    if bolong:
        masalah.append(f"{len(bolong)} jendela bolong")
    if header_beda:
        masalah.append(f"{len(header_beda)} berkas header berbeda")
    if baris_cacat:
        masalah.append(f"{baris_cacat} baris cacat")
    if luar_kotak:
        masalah.append(f"{luar_kotak} titik di luar kotak")
    if masalah:
        print("Perlu ditangani: " + "; ".join(masalah))
        return 2
    print("Tidak ada masalah struktural.")
    print()
    print("Yang MASIH harus diperiksa mata sendiri, bukan oleh skrip ini:")
    print("  - apakah peringkat tahun cocok dengan yang Anda ketahui")
    print("  - apakah puncak bulanannya masuk akal")
    print("Setelah itu baru bangun panel kabupaten-bulan dengan poligon GADM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
