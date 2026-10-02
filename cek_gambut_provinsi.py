"""
Pemeriksaan penyeberangan gambut BPS -> GADM per provinsi (panel telaah ketiga, methods:M3, domain:D8).
Membandingkan jumlah gambut_ha per provinsi GADM (gambut_kabupaten.csv) dengan berkas provinsi Trase
(provinsi Papua baru digabung ke Papua dan Papua Barat seperti di gabung_gambut_nasional.py).
Pakai: python cek_gambut_provinsi.py
"""
import os, re
import pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
norm = lambda s: re.sub(r"[^A-Z]", "", s.upper())
PROV = {"KEPULAUANBANGKABELITUNG": "BANGKABELITUNG", "DAERAHISTIMEWAYOGYAKARTA": "YOGYAKARTA",
        "DAERAHKHUSUSIBUKOTAJAKARTA": "JAKARTARAYA", "PAPUASELATAN": "PAPUA", "PAPUATENGAH": "PAPUA",
        "PAPUAPEGUNUNGAN": "PAPUA", "PAPUABARATDAYA": "PAPUABARAT"}
g = pd.read_csv(os.path.join(D, "gambut_kabupaten.csv"))
p = pd.read_csv(os.path.join(D, "panel_nasional.csv"), usecols=["gid", "provinsi"]).drop_duplicates()
a = g.merge(p, on="gid").assign(pk=lambda x: x.provinsi.map(norm)).groupby("pk").gambut_ha.sum()
t = pd.read_csv(os.path.join(D, "gambut", "spatial-metrics-indonesia-peat_area_province.csv"))
t = t[t.year == 2024].assign(pk=lambda x: x.region.map(norm).replace(PROV)).groupby("pk").peatland_area_hectares.sum()
k = pd.DataFrame({"gadm": a, "trase_provinsi": t}).fillna(0)
k = k[(k.gadm > 0) | (k.trase_provinsi > 0)]
k["selisih_ha"] = k.gadm - k.trase_provinsi
print((k / 1e3).round(1).sort_values("trase_provinsi", ascending=False).to_string())
print(f"total: GADM {k.gadm.sum() / 1e6:.3f} juta ha, Trase provinsi {k.trase_provinsi.sum() / 1e6:.3f} juta ha; "
      f"selisih mutlak terbesar {k.selisih_ha.abs().max():.0f} ha")
