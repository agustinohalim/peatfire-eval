"""
PeatFireBench revisi fase B1 — apakah provinsi berbeda dalam model terbaiknya? (Hasil_Patokan_7, R6)

S1 2020-2025, model dilatih nasional; rf/xgb/mlp dirata-rata atas enam benih. Rujukan klimatologi
terkuat per sasaran (klim_frek1 per kabupaten, klim_frek nasional). Statistik D (bagian pasangan
provinsi berbeda juara) dan H (SB antarprovinsi keunggulan rf - rujukan); nol acak dan nol
bersebelahan (graf ketetanggaan GADM), 200 permutasi; biaya transfer tervalidasi silang genap/ganjil.

Pakai:    python patokan_revisi_provinsi.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_provinsi.log
"""
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "DL_FIRE_NASIONAL")
SEED = 2026
MIN_POS = 40
N_PERM = 200
BIASA = [str(t) for t in range(2020, 2026)]
GENAP, GANJIL = ["2020", "2022", "2024"], ["2021", "2023", "2025"]
RUJUKAN = {"y_kabupaten": "klim_frek1", "y_gabungan": "klim_frek"}
BELAJAR = ["rf", "xgb", "mlp"]

P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
P = P[(P.skema == "S1") & P.tag.isin(BIASA)]
B = pd.read_csv(os.path.join(D, "patokan_benih.csv"), dtype={"tag": str})
B = B[(B.skema == "S1") & B.tag.isin(BIASA)]
kunci = ["sasaran", "tag", "gid", "bulan"]
P = P.merge(B[kunci + [f"{m}_{i}" for m in BELAJAR for i in range(1, 6)]], on=kunci, how="left", validate="one_to_one")
for m in BELAJAR:
    kol = [m] + [f"{m}_{i}" for i in range(1, 6)]
    assert P[kol].notna().all().all(), m
    P[m] = P[kol].mean(axis=1)
print(f"baris {len(P)}; benih 1-5 tergabung untuk {BELAJAR} (rata-rata enam benih)")

# ---- graf ketetanggaan dari batas GADM (titik batas bersama)
W = pd.read_csv(os.path.join(BASE, "DL_FIRE_SV-C2_792597", "gadm41_IDN_2_wkt.csv"))
titik = defaultdict(set)
for g, wkt in zip(W.GID_2, W.WKT):
    for x, y in re.findall(r"(-?\d+\.\d+) (-?\d+\.\d+)", wkt):
        titik[(round(float(x), 4), round(float(y), 4))].add(g)
hit = defaultdict(int)
for gs in titik.values():
    if 1 < len(gs) < 6:
        gs = sorted(gs)
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                hit[(gs[i], gs[j])] += 1
TETANGGA = defaultdict(set)
for (a, b), n in hit.items():
    if n >= 2:
        TETANGGA[a].add(b); TETANGGA[b].add(a)
print(f"graf: {len(W)} kabupaten, {sum(len(v) for v in TETANGGA.values()) // 2} sisi, "
      f"{sum(1 for g in W.GID_2 if not TETANGGA[g])} tanpa tetangga")


def dalam_tahun(g, m, tags):
    v = [ap(x.y, x[m]) for t, x in g.groupby("tag") if t in tags and x.y.sum()]
    return float(np.mean(v)) if v else np.nan


def juara(skor):
    return max(skor, key=lambda m: -np.inf if np.isnan(skor[m]) else skor[m])


def statistik(grup, model, ref, tags=BIASA):
    s = {p: {m: dalam_tahun(g, m, tags) for m in model} for p, g in grup.items()}
    j = {p: juara(s[p]) for p in grup}
    ps = list(grup)
    d = np.mean([j[p] != j[q] for p in ps for q in ps if q != p])
    h = np.nanstd([s[p]["rf"] - s[p][ref] for p in ps], ddof=1)
    return d, h, j


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


def tumbuh(ukuran, kab, rng, cek):
    """wilayah bersebelahan berukuran tertentu dari benih acak; None bila gagal."""
    bebas = set(kab)
    hasil = []
    for n in sorted(ukuran, reverse=True):
        for _ in range(50):
            benih = rng.choice(sorted(bebas))
            w = {benih}; tepi = (TETANGGA[benih] & bebas) - w
            while len(w) < n and tepi:
                x = rng.choice(sorted(tepi)); w.add(x); tepi |= (TETANGGA[x] & bebas); tepi -= w
            if len(w) == n and cek(w):
                break
        else:
            return None
        bebas -= w; hasil.append(w)
    return hasil


rng = np.random.default_rng(SEED)
for sas, ref in RUJUKAN.items():
    d = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d = d[d.ada_positif == 1]
    model = [ref, "persistence", "log_klim_jeda", "fwi_logistik"] + BELAJAR
    pos = d.groupby("provinsi").y.sum()
    prov = sorted(pos[pos >= MIN_POS].index)
    grup = {p: d[d.provinsi == p] for p in prov}
    D_obs, H_obs, j = statistik(grup, model, ref)
    print(f"\n=== {sas}, rujukan {ref}: {len(prov)} provinsi >= {MIN_POS} positif")
    for p in prov:
        g = grup[p]
        print(f"  {p:22s} kab {g.gid.nunique():3d} pos {int(g.y.sum()):4d}  juara {j[p]:14s} rf-ref {dalam_tahun(g, 'rf', BIASA) - dalam_tahun(g, ref, BIASA):+.3f}")
    print(f"  teramati: D = {D_obs:.3f}, H = {H_obs:.3f}; juara: {pd.Series(j).value_counts().to_dict()}")

    ukuran = [grup[p].gid.nunique() for p in prov]
    pos_kab = d.groupby("gid").y.sum()
    cek = lambda w: pos_kab.reindex(list(w)).fillna(0).sum() >= MIN_POS
    kab = sorted(d.gid.unique())
    for nama in ("acak", "bersebelahan"):
        nol = []
        while len(nol) < N_PERM:
            if nama == "acak":
                acak = rng.permutation(kab); w, i = [], 0
                for n in sorted(ukuran, reverse=True):
                    w.append(set(acak[i:i + n])); i += n
                if not all(cek(x) for x in w):
                    continue
            else:
                w = tumbuh(ukuran, kab, rng, cek)
                if w is None:
                    continue
            gp = {f"p{i}": d[d.gid.isin(x)] for i, x in enumerate(w)}
            nol.append(statistik(gp, model, ref)[:2])
        nol = np.array(nol)
        print(f"  nol {nama:12s} D median {np.median(nol[:, 0]):.3f} [p95 {np.percentile(nol[:, 0], 95):.3f}], P = {np.mean(nol[:, 0] >= D_obs):.3f};"
              f"  H median {np.median(nol[:, 1]):.3f} [p95 {np.percentile(nol[:, 1], 95):.3f}], P = {np.mean(nol[:, 1] >= H_obs):.3f}")

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
    print(f"  biaya transfer tervalidasi silang: juara provinsi lain {lain:+.3f} [{np.percentile(bs[:, 0], 2.5):+.3f}, {np.percentile(bs[:, 0], 97.5):+.3f}];"
          f" juara nasional {nas:+.3f} [{np.percentile(bs[:, 1], 2.5):+.3f}, {np.percentile(bs[:, 1], 97.5):+.3f}]")
print("\nselesai.")
