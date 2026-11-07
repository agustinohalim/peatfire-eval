"""
PeatFireBench revisi fase B1, bagian yang tidak sempat dijalankan (Hasil_Patokan_7, penyimpangan).

patokan_revisi_provinsi.py dihentikan saat membangun nol bersebelahan untuk sasaran nasional: wilayah
bersebelahan acak hampir tidak pernah memenuhi syarat >= 40 positif untuk ke-19 provinsi palsu (0 dari 70
percobaan uji, tiga varian pertumbuhan). Skrip ini menjalankan sisa B1 untuk sasaran nasional: biaya
transfer tervalidasi silang genap/ganjil dengan selang 200 resampel kabupaten. Fungsi disalin apa adanya
dari patokan_revisi_provinsi.py; RNG seed 2026 dimulai ulang di sini, jadi tarikan bootstrap berbeda dari
yang akan dipakai skrip asal.

Pakai:    python patokan_revisi_provinsi_biaya.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_provinsi_biaya.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
SEED = 2026
MIN_POS = 40
BIASA = [str(t) for t in range(2020, 2026)]
GENAP, GANJIL = ["2020", "2022", "2024"], ["2021", "2023", "2025"]
BELAJAR = ["rf", "xgb", "mlp"]

P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
P = P[(P.skema == "S1") & P.tag.isin(BIASA)]
B = pd.read_csv(os.path.join(D, "patokan_benih.csv"), dtype={"tag": str})
B = B[(B.skema == "S1") & B.tag.isin(BIASA)]
kunci = ["sasaran", "tag", "gid", "bulan"]
P = P.merge(B[kunci + [f"{m}_{i}" for m in BELAJAR for i in range(1, 6)]], on=kunci, how="left", validate="one_to_one")
for m in BELAJAR:
    P[m] = P[[m] + [f"{m}_{i}" for i in range(1, 6)]].mean(axis=1)


def dalam_tahun(g, m, tags):
    v = [ap(x.y, x[m]) for t, x in g.groupby("tag") if t in tags and x.y.sum()]
    return float(np.mean(v)) if v else np.nan


def juara(skor):
    return max(skor, key=lambda m: -np.inf if np.isnan(skor[m]) else skor[m])


def biaya_silang(grup, model, data_nas):
    out = []
    for pilih, nilai in ((GENAP, GANJIL), (GANJIL, GENAP)):
        sp = {p: {m: dalam_tahun(g, m, pilih) for m in model} for p, g in grup.items()}
        sn = {m: dalam_tahun(data_nas, m, pilih) for m in model}
        jp = {p: juara(sp[p]) for p in grup}; jn = juara(sn)
        sv = {p: {m: dalam_tahun(g, m, nilai) for m in model} for p, g in grup.items()}
        ps = list(grup)
        lain = np.nanmean([sv[p][jp[p]] - sv[p][jp[q]] for p in ps for q in ps if q != p])
        nas = np.nanmean([sv[p][jp[p]] - sv[p][jn] for p in ps])
        out.append((lain, nas))
    return np.mean(out, axis=0)


rng = np.random.default_rng(SEED)
sas, ref = "y_gabungan", "klim_frek"
d = P[P.sasaran == sas]
model = [ref, "persistence", "log_klim_jeda", "fwi_logistik"] + BELAJAR
pos = d.groupby("provinsi").y.sum()
grup = {p: d[d.provinsi == p] for p in sorted(pos[pos >= MIN_POS].index)}
lain, nas = biaya_silang(grup, model, d)
bs = []
for _ in range(200):
    gr = {}
    for p, g in grup.items():
        kk = g.gid.unique(); pilih = rng.choice(kk, len(kk))
        gr[p] = pd.concat([g[g.gid == k] for k in pilih])
    kk = d.gid.unique(); pilih = rng.choice(kk, len(kk))
    dn = pd.concat([d[d.gid == k] for k in pilih])
    bs.append(biaya_silang(gr, model, dn))
bs = np.array(bs)
print(f"=== {sas}, rujukan {ref}: {len(grup)} provinsi")
print(f"  biaya transfer tervalidasi silang: juara provinsi lain {lain:+.3f} [{np.percentile(bs[:, 0], 2.5):+.3f}, {np.percentile(bs[:, 0], 97.5):+.3f}];"
      f" juara nasional {nas:+.3f} [{np.percentile(bs[:, 1], 2.5):+.3f}, {np.percentile(bs[:, 1], 97.5):+.3f}]")
print("\nselesai.")
