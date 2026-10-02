"""
PeatFireBench — analisis kepekaan benih (patokan_benih.csv dari patokan_benih.py).

Untuk tiap benih b (42 = putaran naskah, 1-5 = ulang): rf, xgb, mlp diganti dengan prediksi benih b,
baseline lain tetap. Dilaporkan:
  A. keunggulan nasional dalam-tahun atas klimatologi (S1 2020-2025) per model: rentang antarbenih
  B. ketidaksepakatan juara provinsi dan nol permutasi (100 permutasi per benih), dua sasaran
  C. sama dengan B tetapi pada rata-rata prediksi enam benih (ansambel benih), yang meredam derau benih
  D. ablasi tidak diulang di sini (lihat catatan di log).

Pakai:    python patokan_benih_analisis.py   (DL_FIRE_NASIONAL/penuh/patokan_benih_analisis.log)
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from kelompok_fitur import MODEL, PREDIKSI

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
kunci = ["gid", "tahun", "bulan", "sasaran", "skema", "tag"]
P = pd.read_csv(os.path.join(D, PREDIKSI), dtype={"tag": str})
B = pd.read_csv(os.path.join(D, "patokan_benih.csv"), dtype={"tag": str})
BIASA = [str(t) for t in range(2020, 2026)]
P = P[(P.skema == "S1") & P.tag.isin(BIASA)]
P = P.merge(B.drop(columns=["provinsi", "ada_positif", "y"]), on=kunci, validate="one_to_one")
BENIH = [42, 1, 2, 3, 4, 5]
for m in ("rf", "xgb", "mlp"):
    P[f"{m}_42"] = P[m]
    P[f"{m}_ens"] = P[[f"{m}_{b}" for b in BENIH]].mean(axis=1)
rng = np.random.default_rng(2026)


def dalam(d, m):
    return float(np.mean([ap(g.y, g[m]) for _, g in d.groupby("tag") if g.y.sum()]))


def sepakat(d, prov, label, ganti):
    kol = {m: ganti.get(m, m) for m in MODEL}
    juara = {}
    for p in prov:
        g = d[d.gid.map(label) == p]
        sk = {m: dalam(g, kol[m]) for m in MODEL}
        juara[p] = max(sk, key=sk.get)
    return np.mean([juara[a] != juara[b] for a in prov for b in prov if a != b]), juara


for sas in ("y_kabupaten", "y_gabungan"):
    d = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d = d[d.ada_positif == 1]
    print(f"\n=== {sas}")
    print("A. keunggulan nasional atas klimatologi, per benih")
    k = dalam(d, "klimatologi")
    for m in ("rf", "xgb", "mlp"):
        v = [dalam(d, f"{m}_{b}") - k for b in BENIH]
        print(f"  {m:4s} " + "  ".join(f"{b}:{x:+.3f}" for b, x in zip(BENIH, v))
              + f"   rentang {min(v):+.3f}..{max(v):+.3f}   ansambel {dalam(d, m + '_ens') - k:+.3f}")
    pos = d.groupby("provinsi").y.sum()
    prov = sorted(pos[pos >= 40].index)
    d = d[d.provinsi.isin(prov)]
    peta = d.groupby("gid").provinsi.first()
    print(f"B/C. ketidaksepakatan juara di {len(prov)} provinsi, lawan nol permutasi (100 per baris)")
    for nama in [str(b) for b in BENIH] + ["ens"]:
        ganti = {m: f"{m}_{nama}" for m in ("rf", "xgb", "mlp")}
        amat, juara = sepakat(d, prov, peta, ganti)
        nol = [sepakat(d, prov, pd.Series(rng.permutation(peta.to_numpy()), index=peta.index), ganti)[0]
               for _ in range(100)]
        hit = pd.Series(juara).value_counts().to_dict()
        print(f"  benih {nama:4s} teramati {amat * 100:3.0f}%  nol median {np.median(nol) * 100:3.0f}% "
              f"[{np.percentile(nol, 2.5) * 100:.0f}%, {np.percentile(nol, 97.5) * 100:.0f}%]  "
              f"P(nol >= teramati) {np.mean(np.array(nol) >= amat):.2f}  juara {hit}")
print("\nselesai.")
