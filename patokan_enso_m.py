"""
Revisi fase M (Hasil_Patokan_18), M3: fase ENSO/IOD yang diketahui sebelum musim kebakaran, per tahun uji,
dengan BSS per tahun GB atas EF-L dan EF-L atas kalender (deskriptif, tanpa aturan).
  ONI musim AMJ (berakhir Juni) dari oni.ascii.txt NOAA: El Nino >= +0,5, La Nina <= -0,5, selain itu netral.
  DMI rata-rata Mei-Juni dari dmi.had.long.data (HadISST, NOAA PSL): positif >= +0,4, negatif <= -0,4.
  ONI ASO dicetak untuk memeriksa nilai yang disebut naskah.
Nilai per tahun dibaca dari log, bukan diketik:
  DL_FIRE_MODIS/patokan_modis_k_{kab,nas}.log          prakiraan maju MODIS 2008-2019 (fase K, lantai 2)
  DL_FIRE_NASIONAL/penuh/patokan_tingkat_l_{kab,nas}.log  VIIRS 2020-2025 (fase L)

Pakai:    python -u patokan_enso_m.py > DL_FIRE_NASIONAL/penuh/patokan_enso_m.log
"""
import os
import re

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
IDX = next(x for x in (os.path.join(D, "data"), os.path.join(D, "DL_FIRE_SV-C2_792597"))
           if os.path.exists(os.path.join(x, "oni.ascii.txt")))

oni = {}
for ln in open(os.path.join(IDX, "oni.ascii.txt")):
    p = ln.split()
    if len(p) == 4 and p[1].isdigit():
        oni[(p[0], int(p[1]))] = float(p[3])
dmi = {}
for ln in open(os.path.join(IDX, "dmi.had.long.data")):
    p = ln.split()
    if len(p) == 13 and p[0].isdigit():
        v = [float(x) for x in p[1:]]
        if v[4] > -99 and v[5] > -99:
            dmi[int(p[0])] = (v[4] + v[5]) / 2


def fase_oni(x):
    return "El Nino" if x >= 0.5 else ("La Nina" if x <= -0.5 else "netral")


def fase_dmi(x):
    return "positif" if x >= 0.4 else ("negatif" if x <= -0.4 else "netral")


def modis(nama):
    teks = open(os.path.join(D, "DL_FIRE_MODIS", nama), encoding="utf-8").read()
    prev = {int(a): float(b) for a, b in re.findall(r"R (\d{4}): .*?prevalensi ([\d.]+)%", teks)}
    out = {}
    for kunci, judul in (("gb/efl", "gb atas efl"), ("efl/kal", "efl atas kalender")):
        blok = teks[teks.index("    " + judul + "\n"):]
        baris = re.search(r"per tahun (.*)", blok).group(1)
        out[kunci] = {int(a): float(b) for a, b in re.findall(r"(\d{4}):([+-][\d.]+)", baris)}
    return prev, out


def viirs(nama):
    teks = open(os.path.join(D, "DL_FIRE_NASIONAL", "penuh", nama), encoding="utf-8").read()
    kepala = re.search(r"prevalensi per tahun (.*)", teks).group(1)
    prev = {int(a): float(b) for a, b in re.findall(r"(\d{4}) ([\d.]+)%", kepala)}
    out, acu = {}, None
    for ln in teks.splitlines():
        m = re.match(r"\s+-- atas (\S+)", ln)
        if m:
            acu = m.group(1)
            continue
        m = re.match(r"\s+(\S+)\s+\(a\).*per tahun (.*)", ln)
        if m and acu:
            nilai = [float(v) for v in m.group(2).split()]
            if m.group(1) == "xgb" and acu == "ef1_tingkat":
                out["gb/efl"] = dict(zip(range(2020, 2026), nilai))
            if m.group(1) == "ef1_tingkat" and acu.startswith("klim_frek"):
                out["efl/kal"] = dict(zip(range(2020, 2026), nilai))
    return prev, out


print("Tahun  ONI AMJ  fase      ONI ASO  DMI MJ  fase IOD | per kab: prev  GB/EF-L  EF-L/kal | nasional: prev  GB/EF-L  EF-L/kal")
BAR = []
for sumber, fk, fn in (("MODIS maju", modis("patokan_modis_k_kab.log"), modis("patokan_modis_k_nas.log")),
                       ("VIIRS", viirs("patokan_tingkat_l_kab.log"), viirs("patokan_tingkat_l_nas.log"))):
    print(f"-- {sumber}")
    for th in sorted(fk[1]["gb/efl"]):
        o, oa, d = oni[("AMJ", th)], oni[("ASO", th)], dmi.get(th, np.nan)
        r = dict(sumber=sumber, th=th, fo=fase_oni(o), fd=fase_dmi(d),
                 k_gb=fk[1]["gb/efl"][th], k_efl=fk[1]["efl/kal"][th], n_gb=fn[1]["gb/efl"][th], n_efl=fn[1]["efl/kal"][th])
        BAR.append(r)
        print(f"{th}  {o:+.2f}   {r['fo']:<8}  {oa:+.2f}    {d:+.2f}  {r['fd']:<8} |  {fk[0][th]:5.2f}%  {r['k_gb']:+.3f}   "
              f"{r['k_efl']:+.3f}   |  {fn[0][th]:5.2f}%  {r['n_gb']:+.3f}   {r['n_efl']:+.3f}")

print("\nRata-rata BSS per fase (n tahun)")
for sumber in ("MODIS maju", "VIIRS", "keduanya"):
    for kol, label in (("fo", "ONI AMJ"), ("fd", "DMI MJ")):
        bag = [r for r in BAR if sumber == "keduanya" or r["sumber"] == sumber]
        for f in sorted({r[kol] for r in bag}):
            x = [r for r in bag if r[kol] == f]
            mu = {k: np.mean([r[k] for r in x]) for k in ("k_gb", "k_efl", "n_gb", "n_efl")}
            print(f"  {sumber:<10} {label:<8} {f:<8} n {len(x):>2}: per kab GB/EF-L {mu['k_gb']:+.3f} EF-L/kal {mu['k_efl']:+.3f}"
                  f" | nasional GB/EF-L {mu['n_gb']:+.3f} EF-L/kal {mu['n_efl']:+.3f}  tahun {', '.join(str(r['th']) for r in x)}")
print("selesai.")
