#!/usr/bin/env python3
"""Gabungkan ONI, DMI, dan jeda titik panas ke panel nasional.

Langkah 3 sesudah unduhan dan pembangunan panel. Menyiapkan peubah penjelas,
**bukan** menjalankan model. Bagian I `Artikel_2_Rencana.md` menaruh percobaan
pada Apr-Jun 2027, dan urutan itu ada sebabnya: kumpulan data yang bersih dan
protokol yang benar yang menentukan diterima atau tidaknya artikel, bukan
arsitektur.

Konvensi disalin dari kode Artikel 1 supaya sebanding, bukan sekadar mirip:

  - ONI dari `oni.ascii.txt`, kolom keempat, musim tiga bulan dipetakan ke
    bulan terakhirnya lewat indeks SEASONS — sama dengan `percobaan.py`
  - DMI dari `dmi.had.long.data`, format NOAA PSL 13 kolom, nilai di bawah
    -90 dianggap kosong — sama dengan `percobaan3_dmi_ambang.py`
  - Jeda titik panas 1, 2, 3, dan 12 bulan per kabupaten — sama dengan
    `percobaan.py`
  - ONI jeda 1 sampai 6; DMI jeda 1 sampai 3 seperti varian kedelapan pada
    naskah bagian 4.1
  - Suku bulan siklik sin dan cos

YANG SENGAJA TIDAK DIKERJAKAN DI SINI: peubah sasaran.

Artikel 1 memakai ambang persentil ke-90 atas **seluruh** sel kabupaten-bulan
yang digabung. Untuk 14 kabupaten Kalimantan Barat yang sebanding, itu wajar.
Untuk 502 kabupaten seluruh Indonesia, ambang gabungan berisiko merosot
menjadi penanda identitas kabupaten: kabupaten kota yang kecil hampir tak
pernah melampauinya, kabupaten besar yang rawan hampir selalu. Model lalu
belajar "ini kabupaten mana", bukan "kapan kebakaran parah".

Karena itu skrip ini hanya **mengukur** persoalannya di bagian akhir, lalu
berhenti. Pilihan ambang adalah keputusan rancangan Artikel 2 yang harus
ditulis sadar, bukan diwarisi diam-diam.

Pemakaian:
    python gabung_indeks_nasional.py \\
        DL_FIRE_NASIONAL/panel_nasional.csv \\
        DL_FIRE_SV-C2_792597/oni.ascii.txt \\
        DL_FIRE_SV-C2_792597/dmi.had.long.data \\
        DL_FIRE_NASIONAL/panel_nasional_fitur.csv
"""

import sys

import numpy as np
import pandas as pd

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def muat_oni(path):
    """Sama dengan percobaan.py: kolom keempat, musim ke bulan pusat."""
    baris = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) < 4 or p[0] not in SEASONS:
                continue
            # musim ditaruh pada bulan terakhirnya (lihat percobaan.py, revisi 1 Okt 2026)
            th, bl = int(p[1]), SEASONS.index(p[0]) + 2
            if bl > 12:
                th, bl = th + 1, bl - 12
            baris.append((th, bl, float(p[3])))
    oni = pd.DataFrame(baris, columns=["tahun", "bulan_ke", "oni"])
    oni["periode"] = pd.PeriodIndex(
        oni["tahun"].astype(str) + "-" + oni["bulan_ke"].astype(str).str.zfill(2),
        freq="M",
    )
    return dict(zip(oni["periode"], oni["oni"]))


def muat_dmi(path):
    """Sama dengan percobaan3_dmi_ambang.py: NOAA PSL 13 kolom, -90 kosong."""
    nilai = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 13:
                continue
            try:
                th = int(p[0])
            except ValueError:
                continue
            if not (1870 <= th <= 2030):
                continue
            for m, v in enumerate(p[1:], 1):
                x = float(v)
                if x < -90:
                    continue
                nilai[pd.Period(f"{th}-{m:02d}", freq="M")] = x
    return nilai


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 1
    panel_path, oni_path, dmi_path, out_path = sys.argv[1:5]

    df = pd.read_csv(panel_path)
    df["periode"] = pd.PeriodIndex(df["bulan"], freq="M")
    print(f"panel dimuat  : {len(df):,} baris, "
          f"{df['gid'].nunique()} kabupaten, {df['periode'].nunique()} bulan")

    oni_map = muat_oni(oni_path)
    dmi_map = muat_dmi(dmi_path)
    print(f"ONI dimuat    : {len(oni_map)} bulan")
    print(f"DMI dimuat    : {len(dmi_map)} bulan")

    df = df.sort_values(["gid", "periode"]).reset_index(drop=True)
    g = df.groupby("gid", observed=True)["titik_panas"]
    for lag in (1, 2, 3, 12):
        df[f"tp_lag{lag}"] = g.shift(lag)
    for lag in range(1, 7):
        df[f"oni_lag{lag}"] = (df["periode"] - lag).map(oni_map)
    for lag in range(1, 4):
        df[f"dmi_lag{lag}"] = (df["periode"] - lag).map(dmi_map)
    df["bulan_sin"] = np.sin(2 * np.pi * df["bulan_ke"] / 12)
    df["bulan_cos"] = np.cos(2 * np.pi * df["bulan_ke"] / 12)

    # Buang 12 bulan pertama tiap kabupaten, sama dengan Artikel 1.
    sebelum = len(df)
    df = df.dropna(subset=["tp_lag12"]).reset_index(drop=True)
    print(f"dibuang       : {sebelum - len(df):,} baris tanpa tp_lag12 "
          f"(12 bulan pertama tiap kabupaten)")

    kosong = {k: int(df[k].isna().sum()) for k in df.columns if df[k].isna().any()}
    print(f"kolom bernilai kosong: {kosong if kosong else 'tidak ada'}")

    df.drop(columns=["periode"]).to_csv(out_path, index=False)
    print(f"ditulis       : {out_path}  ({len(df):,} baris, {len(df.columns) - 1} kolom)")

    # ------------------------------------------------------------------
    # Diagnostik ambang. Mengukur, tidak memutuskan.
    # ------------------------------------------------------------------
    print()
    print("=" * 74)
    print("DIAGNOSTIK AMBANG — dibaca sebelum memilih peubah sasaran")
    print("=" * 74)

    ambang_gabung = df["titik_panas"].quantile(0.90)
    print(f"\nAmbang persentil ke-90 gabungan, cara Artikel 1: {ambang_gabung:.1f} titik")

    per_kab = df.groupby("gid", observed=True)["titik_panas"]
    lampaui = per_kab.apply(lambda s: (s > ambang_gabung).mean())
    n_kab = len(lampaui)
    nol = int((lampaui == 0).sum())
    hampir_nol = int((lampaui < 0.01).sum())
    di_atas_sepertiga = int((lampaui > 1 / 3).sum())

    print(f"  kabupaten                          : {n_kab}")
    print(f"  yang TIDAK PERNAH melampaui ambang : {nol}  ({nol / n_kab * 100:.1f}%)")
    print(f"  yang melampaui < 1% bulan          : {hampir_nol}  ({hampir_nol / n_kab * 100:.1f}%)")
    print(f"  yang melampaui > 33% bulan         : {di_atas_sepertiga}")
    print(f"  prevalensi positif keseluruhan     : {df['titik_panas'].gt(ambang_gabung).mean() * 100:.1f}%")

    print("\n  Sepuluh kabupaten dengan prevalensi tertinggi di bawah ambang gabungan:")
    for gid, v in lampaui.sort_values(ascending=False).head(10).items():
        b = df[df["gid"] == gid].iloc[0]
        print(f"    {b['provinsi']:<22} {b['kabupaten']:<24} {v * 100:5.1f}%")

    ambang_kab = per_kab.transform(lambda s: s.quantile(0.90))
    y_kab = (df["titik_panas"] > ambang_kab).astype(int)
    print(f"\nAlternatif: persentil ke-90 PER KABUPATEN")
    print(f"  prevalensi positif keseluruhan     : {y_kab.mean() * 100:.1f}%")
    kab_tanpa_positif = int(
        (y_kab.groupby(df["gid"], observed=True).sum() == 0).sum()
    )
    print(f"  kabupaten tanpa satu pun positif   : {kab_tanpa_positif}")

    print()
    print("Bacaannya begini. Kalau angka 'tidak pernah melampaui' itu besar, ambang")
    print("gabungan membuat peubah sasaran sebagian menjadi penanda identitas")
    print("kabupaten, dan model apa pun akan tampak hebat hanya karena menghafal")
    print("kabupaten mana yang rawan. Ambang per kabupaten menghindarinya, tetapi")
    print("menukar artinya: 'parah untuk kabupaten ini', bukan 'parah secara nasional'.")
    print()
    print("Keduanya sah. Yang tidak sah adalah memilih tanpa menyatakan alasannya.")
    print("Catat pilihannya di Artikel_2_Rencana.md sebelum percobaan dijalankan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
