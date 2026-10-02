"""
Menggabungkan luas gambut per kabupaten ke panel fitur nasional PeatFireBench.

Sumber: Peta Lahan Gambut Indonesia skala 1:50.000, edisi Desember 2019, Balai Besar Sumber Daya
Lahan Pertanian, Kementerian Pertanian (Anda et al., 2021), dalam bentuk luas per kabupaten
yang diterbitkan Trase (spatial-metrics-indonesia-peat_area_kabupaten.csv, rilis 18 September
2026, CC BY 4.0; diunduh 1 Oktober 2026 dari resources.trase.earth). Total 13,39 juta ha di 514
kabupaten/kota BPS; nilainya sama di semua tahun 2015-2024 (peta statis), jadi tahun 2024 dipakai.

Masukan : DL_FIRE_NASIONAL/panel_nasional_fitur_lengkap.csv
          DL_FIRE_NASIONAL/statis_kabupaten.csv      (luas_km2 satuan GADM, dari Earth Engine)
          DL_FIRE_NASIONAL/gambut/spatial-metrics-indonesia-peat_area_kabupaten.csv
Keluaran: DL_FIRE_NASIONAL/panel_nasional_fitur_gambut.csv   (berkas lengkap tidak diubah)
          DL_FIRE_NASIONAL/gambut_kabupaten.csv              (gid, gambut_ha, gambut_fraksi)

Penyeberangan BPS -> GADM. Trase memakai 514 kabupaten BPS sekarang; GADM 4.1 level 2 memakai
batas sebelum pemekaran 2013 (498 satuan daratan). Dicocokkan pada provinsi + nama + jenis
(kabupaten atau kota, karena "Kota Pontianak" dan "Pontianak" berbeda). Kabupaten hasil pemekaran
sesudah batas GADM dijumlahkan ke kabupaten induknya (daftar INDUK di bawah, dari undang-undang
pembentukannya), begitu pula provinsi Papua baru ke Papua dan Papua Barat. Pemeriksaan: setiap
kabupaten BPS harus mendapat satu gid dan setiap satuan GADM daratan minimal satu kabupaten BPS,
kecuali Danau Limboto (satuan danau di GADM).

gambut_fraksi = gambut_ha / luas GADM, dipotong ke 1 karena luas BPS dan GADM tidak sama persis.
Fitur statis, tanpa jeda: peta gambut tidak berubah menurut tahun uji, jadi tidak bocor.

Pakai:  python gabung_gambut_nasional.py
"""

import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
TRASE = os.path.join(D, "gambut", "spatial-metrics-indonesia-peat_area_kabupaten.csv")

PROV = {"KEPULAUANBANGKABELITUNG": "BANGKABELITUNG", "DAERAHISTIMEWAYOGYAKARTA": "YOGYAKARTA",
        "DAERAHKHUSUSIBUKOTAJAKARTA": "JAKARTARAYA", "PAPUASELATAN": "PAPUA", "PAPUATENGAH": "PAPUA",
        "PAPUAPEGUNUNGAN": "PAPUA", "PAPUABARATDAYA": "PAPUABARAT"}
# Nama BPS -> nama GADM: ganti nama, ejaan lain, dan pemekaran sesudah batas GADM (ke induk).
INDUK = {
    "ADMINISTRASI KEPULAUAN SERIBU": "KepulauanSeribu", "KEP. SIAU TAGULANDANG BIARO": "SiauTagulandangBiaro",
    "KOTABARU": "KotaBaru", "KOTA KOTAMOBAGU": "Kotamobagu", "KOTA TARAKAN": "Tarakan",
    "PAKPAK BHARAT": "PakpakBarat", "TOBA": "TobaSamosir", "KEPULAUAN TANIMBAR": "MalukuTenggaraBarat",
    "TANJUNG JABUNG BARAT": "TanjungJabungB", "TANJUNG JABUNG TIMUR": "TanjungJabungT",
    "MEMPAWAH": "Pontianak", "PASANGKAYU": "MamujuUtara",
    # pemekaran 2012-2013
    "BANGGAI LAUT": "BanggaiKepulauan", "MOROWALI UTARA": "Morowali", "MAMUJU TENGAH": "Mamuju",
    "BUTON SELATAN": "Buton", "BUTON TENGAH": "Buton", "KOLAKA TIMUR": "Kolaka",
    "KONAWE KEPULAUAN": "Konawe", "MUNA BARAT": "Muna", "MUSI RAWAS UTARA": "MusiRawas",
    "PENUKAL ABAB LEMATANG ILIR": "MuaraEnim", "MAHAKAM ULU": "KutaiBarat", "MALAKA": "Belu",
    "MANOKWARI SELATAN": "Manokwari", "PEGUNUNGAN ARFAK": "Manokwari", "PANGANDARAN": "Ciamis",
    "PESISIR BARAT": "LampungBarat", "PULAU TALIABU": "KepulauanSula",
}


def norm(s):
    return re.sub(r"[^A-Z]", "", s.upper())


p = pd.read_csv(os.path.join(D, "panel_nasional.csv"), usecols=["gid", "provinsi", "kabupaten", "tipe"]).drop_duplicates()
p = p[p["tipe"] != "WaterBody"].copy()
p["kota"] = p["tipe"].eq("Kota")
p["pk"], p["k"] = p["provinsi"].map(norm), p["kabupaten"].map(norm)
p.loc[p["kota"], "k"] = p.loc[p["kota"], "k"].str.replace("^KOTA", "", regex=True)   # "KotaPontianak"

t = pd.read_csv(TRASE)
assert t.groupby("region_trase_id")["peatland_area_hectares"].nunique().max() == 1
t = t[t["year"] == 2024].copy()
assert len(t) == 514
t["pk"] = t["parent_region"].map(norm).replace(PROV)
t["kota"] = t["region"].str.startswith("KOTA ")
t["k"] = t["region"].str.replace("^KOTA (ADMINISTRASI )?", "", regex=True).map(norm)
m = t.merge(p[["gid", "pk", "k", "kota"]], on=["pk", "k", "kota"], how="left", validate="many_to_one")
ganti = m["gid"].isna()
lookup = p.set_index(["pk", "kabupaten"])["gid"]
m.loc[ganti, "gid"] = [lookup.get((pk, INDUK.get(r, "?"))) for pk, r in zip(m.loc[ganti, "pk"], m.loc[ganti, "region"])]
assert m["gid"].notna().all(), m.loc[m["gid"].isna(), ["region", "parent_region"]]
tanpa = sorted(set(p["kabupaten"][~p["gid"].isin(m["gid"])]))
assert tanpa == ["DanauLimboto"], tanpa

g = m.groupby("gid")["peatland_area_hectares"].sum().rename("gambut_ha")
s = pd.read_csv(os.path.join(D, "statis_kabupaten.csv")).rename(columns={"GID_2": "gid"}).set_index("gid")
g = g.reindex(p["gid"]).fillna(0.0).to_frame()
g["gambut_fraksi"] = (g["gambut_ha"] / (s["luas_km2"].reindex(g.index) * 100)).clip(upper=1.0)
g.index.name = "gid"
lebih = (g["gambut_ha"] / (s["luas_km2"].reindex(g.index) * 100) > 1).sum()
print(f"total gambut {g.gambut_ha.sum() / 1e6:.2f} juta ha; {(g.gambut_ha > 0).sum()} satuan bergambut; "
      f"{lebih} satuan dipotong ke fraksi 1")
print(g.sort_values("gambut_fraksi", ascending=False).head(8).round(3).to_string())
g.reset_index().to_csv(os.path.join(D, "gambut_kabupaten.csv"), index=False)

f = pd.read_csv(os.path.join(D, "panel_nasional_fitur_lengkap.csv"))
out = f.merge(g[["gambut_fraksi"]].reset_index(), on="gid", how="left", validate="many_to_one")
assert len(out) == len(f) and out["gambut_fraksi"].notna().all()
out.to_csv(os.path.join(D, "panel_nasional_fitur_gambut.csv"), index=False)
r = out.groupby("gid").agg(g=("gambut_fraksi", "first"), y=("y_kabupaten", "mean"), tp=("titik_panas", "sum"))
print(f"{len(out)} baris; Spearman gambut_fraksi vs jumlah titik panas per kabupaten: "
      f"{r['g'].corr(r['tp'], method='spearman'):.3f}")
