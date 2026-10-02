"""
Artikel 2 (PeatFireBench) — uji keberatan panel telaah (Telaah/PeatFireBench_2026-10-01).

  A. Apakah pembalikan tahun ekstrem artefak metrik? Untuk tiap tahun uji (S1 2019-2025, S2 2015,
     2014, 2019), pada himpunan kabupaten yang SAMA (411 kabupaten ber-positif) untuk kedua sasaran:
       - selisih AUC-PR mentah GB - klimatologi;
       - selisih ternormalisasi ruang sisa: (AP_GB - AP_klim) / (1 - AP_klim), bagian dari ruang di
         atas klimatologi yang direbut GB;
       - prevalensi dan AP klimatologi.
     Lalu kontras tahun ekstrem (S2 2015, 2019 dan S1 2023) lawan tahun tenang (S1 2020-2022, 2024,
     2025) per sasaran, dengan selang 95 % bootstrap kabupaten berpasangan (undian sama untuk semua
     tahun dan kedua sasaran), dan selisih kontras antar-sasaran.
  B. Nol permutasi untuk ketidaksepakatan juara provinsi: kabupaten diacak ke "provinsi" palsu
     berukuran sama (200 permutasi); bila ketidaksepakatan teramati setara nol, perbedaan juara antar
     provinsi adalah derau sampel, bukan perbedaan antar-provinsi.

Pakai:    python patokan_uji_panel.py
Keluaran: layar (simpan ke DL_FIRE_NASIONAL/oni_akhir/patokan_uji_panel.log)
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
from kelompok_fitur import MODEL, PREDIKSI
P = pd.read_csv(os.path.join(D, PREDIKSI), dtype={"tag": str})
P = P[P.ada_positif == 1]                                  # 411 kabupaten yang sama untuk kedua sasaran
EKSTREM = [("S2", "2015"), ("S2", "2019"), ("S1", "2023")]
TENANG = [("S1", t) for t in ("2020", "2021", "2022", "2024", "2025")]
rng = np.random.default_rng(2026)


def ukur(g):
    a, k = ap(g.y, g.xgb), ap(g.y, g.klimatologi)
    return a - k, (a - k) / (1 - k), g.y.mean(), k


print("A. Per tahun, 411 kabupaten yang sama")
print(f"  {'sasaran':10s}{'tahun':9s}{'prev':>7s}{'AP klim':>9s}{'GB-klim':>9s}{'ruang sisa':>11s}")
blok = {}
for s in ("y_kabupaten", "y_gabungan"):
    for sk, t in TENANG + EKSTREM + [("S2", "2014"), ("S1", "2019")]:
        g = P[(P.sasaran == s) & (P.skema == sk) & (P.tag == t)]
        blok[(s, sk, t)] = g
        m, r, p, k = ukur(g)
        print(f"  {s:10s}{sk + ' ' + t:9s}{p * 100:6.1f}%{k:9.3f}{m:+9.3f}{r:+11.3f}")

kab = P.gid.unique()
idx = {key: {k: np.where(g.gid.to_numpy() == k)[0] for k in kab} for key, g in blok.items()}


def kontras(pilih, s, j):
    v = {}
    for nama, kel in (("ekstrem", EKSTREM), ("tenang", TENANG)):
        x = []
        for sk, t in kel:
            g = blok[(s, sk, t)]
            ii = np.concatenate([idx[(s, sk, t)][k] for k in pilih]) if pilih is not None else np.arange(len(g))
            y, a, k = g.y.to_numpy()[ii], g.xgb.to_numpy()[ii], g.klimatologi.to_numpy()[ii]
            if y.sum() == 0:
                continue
            ga, gk = ap(y, a), ap(y, k)
            x.append((ga - gk) if j == "mentah" else (ga - gk) / (1 - gk))
        v[nama] = np.mean(x)
    return v["ekstrem"] - v["tenang"]


print("\n   Kontras ekstrem - tenang [95 %, bootstrap kabupaten berpasangan, 300 ulangan]")
for j in ("mentah", "ruang sisa"):
    titik = {s: kontras(None, s, j) for s in ("y_kabupaten", "y_gabungan")}
    bs = []
    for _ in range(300):
        pilih = rng.choice(kab, len(kab))
        bs.append([kontras(pilih, s, j) for s in ("y_kabupaten", "y_gabungan")])
    bs = np.array(bs)
    for i, s in enumerate(("y_kabupaten", "y_gabungan")):
        print(f"  {j:10s} {s:12s} {titik[s]:+.3f} [{np.percentile(bs[:, i], 2.5):+.3f}, {np.percentile(bs[:, i], 97.5):+.3f}]")
    dif = bs[:, 0] - bs[:, 1]
    print(f"  {j:10s} selisih antar-sasaran {titik['y_kabupaten'] - titik['y_gabungan']:+.3f} "
          f"[{np.percentile(dif, 2.5):+.3f}, {np.percentile(dif, 97.5):+.3f}]")

# ---------------------------------------------------------------- B
BIASA = [str(t) for t in range(2020, 2026)]
Q = pd.read_csv(os.path.join(D, PREDIKSI), dtype={"tag": str})
print("\nB. Ketidaksepakatan juara provinsi lawan nol permutasi (provinsi palsu berukuran sama)")
for s, min_pos in (("y_kabupaten", 40), ("y_gabungan", 40)):
    d = Q[(Q.sasaran == s) & (Q.skema == "S1") & Q.tag.isin(BIASA)]
    if s == "y_kabupaten":
        d = d[d.ada_positif == 1]
    pos = d.groupby("provinsi").y.sum()
    prov = sorted(pos[pos >= min_pos].index)
    d = d[d.provinsi.isin(prov)]
    peta = d.groupby("gid").provinsi.first()

    def sepakat(label):
        juara = {}
        for p in prov:
            g = d[d.gid.map(label) == p]
            sk = {m: np.mean([ap(x.y, x[m]) for _, x in g.groupby("tag") if x.y.sum()]) for m in MODEL}
            juara[p] = max(sk, key=sk.get)
        return np.mean([juara[a] != juara[b] for a in prov for b in prov if a != b])

    amat = sepakat(peta)
    nol = []
    for _ in range(200):
        acak = pd.Series(rng.permutation(peta.to_numpy()), index=peta.index)
        nol.append(sepakat(acak))
    print(f"  {s:12s} teramati {amat * 100:.0f}%  nol median {np.median(nol) * 100:.0f}% "
          f"[{np.percentile(nol, 2.5) * 100:.0f}%, {np.percentile(nol, 97.5) * 100:.0f}%]  "
          f"P(nol >= teramati) {np.mean(np.array(nol) >= amat):.2f}")
