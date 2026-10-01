"""
Artikel 2 (PeatFireBench) — apakah perbandingan model di satu provinsi bisa dipakai di provinsi lain?

Dari DL_FIRE_NASIONAL/patokan_prediksi_lengkap.csv, S1 kronologis 2020-2025, model dilatih nasional.
Provinsi dinilai bila punya sedikitnya MIN_POS positif pada 2020-2025 untuk sasaran itu.

  A. Satu provinsi: keunggulan dalam-tahun gradient boosting atas klimatologi per provinsi
     [95 %, bootstrap kabupaten provinsi itu, 1.000 ulangan, undian sama untuk tiap tahun].
     Berapa provinsi yang selangnya tidak memuat nol?
  B. Memilih model dari provinsi lain: juara tiap provinsi (AUC-PR dalam-tahun, delapan model),
     penyesalan bila memakai juara provinsi lain, rerata atas semua pasangan berurutan, dan bagian
     pasangan yang juaranya berbeda; selang 95 % dari 1.000 resampel kabupaten di dalam tiap provinsi
     (seluruh prosedur diulang).

Pakai:    python patokan_provinsi.py
Keluaran: layar (simpan ke DL_FIRE_NASIONAL/oni_akhir/patokan_provinsi.log)
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
P = pd.read_csv(os.path.join(D, "patokan_prediksi_lengkap.csv"), dtype={"tag": str})
MODEL = ["klimatologi", "persistence", "seasonal_naive", "rasio", "logistik", "rf", "xgb", "mlp"]
BIASA = [str(t) for t in range(2020, 2026)]
MIN_POS = 40
rng = np.random.default_rng(2026)


def dalam_tahun(g, m):
    v = [ap(x.y, x[m]) for _, x in g.groupby("tag") if x.y.sum()]
    return float(np.mean(v)) if v else np.nan


def skor_prov(g):
    return {m: dalam_tahun(g, m) for m in MODEL}


def resampel(g):
    kab = g.gid.unique()
    pilih = rng.choice(kab, len(kab))
    return pd.concat([g[g.gid == k] for k in pilih])


for sas in ("y_kabupaten", "y_gabungan"):
    d = P[(P.sasaran == sas) & (P.skema == "S1") & P.tag.isin(BIASA)]
    if sas == "y_kabupaten":
        d = d[d.ada_positif == 1]
    pos = d.groupby("provinsi").y.sum()
    prov = sorted(pos[pos >= MIN_POS].index)
    print(f"\n=== {sas}: {len(prov)} provinsi dengan >= {MIN_POS} positif (2020-2025)")
    grup = {p: d[d.provinsi == p] for p in prov}

    print("A. Keunggulan GB atas klimatologi per provinsi [95 %, bootstrap kabupaten]")
    tegas = 0
    for p in prov:
        g = grup[p]
        nilai = dalam_tahun(g, "xgb") - dalam_tahun(g, "klimatologi")
        bs = [dalam_tahun(x, "xgb") - dalam_tahun(x, "klimatologi") for x in (resampel(g) for _ in range(300))]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        tegas += lo > 0 or hi < 0
        print(f"  {p:22s} positif {int(g.y.sum()):4d}  kabupaten {g.gid.nunique():3d}  {nilai:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"  selang tidak memuat nol: {tegas} dari {len(prov)} provinsi")

    def prosedur(gr):
        s = {p: skor_prov(gr[p]) for p in prov}
        juara = {p: max(s[p], key=lambda m: -np.inf if np.isnan(s[p][m]) else s[p][m]) for p in prov}
        r = [s[p][juara[p]] - s[p][juara[q]] for p in prov for q in prov if q != p]
        salah = np.mean([juara[p] != juara[q] for p in prov for q in prov if q != p])
        return np.nanmean(r), salah, juara

    r, salah, juara = prosedur(grup)
    bs = np.array([prosedur({p: resampel(grup[p]) for p in prov})[:2] for _ in range(200)])
    hit = pd.Series(juara).value_counts().to_dict()
    print(f"B. Penyesalan memilih juara provinsi lain: {r:.3f} [{np.percentile(bs[:, 0], 2.5):.3f}, {np.percentile(bs[:, 0], 97.5):.3f}]; "
          f"juara berbeda pada {salah * 100:.0f}% pasangan [{np.percentile(bs[:, 1], 2.5) * 100:.0f}%, {np.percentile(bs[:, 1], 97.5) * 100:.0f}%]")
    print(f"   juara per provinsi: {hit}")
