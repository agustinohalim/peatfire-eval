"""
Naskah gabungan Kalimantan — analisis tingkat keputusan dan uji keberatan panel telaah (1 Okt 2026).

Dari DL_FIRE_NASIONAL/prediksi_kalimantan.csv (kalimantan_prediksi.py), skema rolling 2019-2025.

  A. Anggaran peringatan tetap. Tiap bulan uji, badan penanggulangan bencana menandai k kabupaten
     dengan skor tertinggi di seluruh Kalimantan (k = 5 dan 10 dari 55). Dilaporkan: bagian
     kabupaten-bulan parah yang tertangkap (hit rate) per model dan per sasaran, dengan selang 95 %
     bootstrap bulan (2.000 ulangan).
  B. Biaya memilih model dari provinsi lain. Untuk tiap provinsi P yang dapat ditafsirkan, model
     terbaik (AUC-PR) dipilih di provinsi Q != P, lalu dinilai di P. Penyesalan = AUC-PR model
     terbaik di P - AUC-PR model pilihan Q di P; rata-rata atas semua pasangan (P, Q). Sama untuk
     memilih dari peringkat seluruh Kalimantan.
  C. Uji rancu prevalensi. Pada sasaran gabungan, tiap provinsi disubsampel negatifnya sampai
     prevalensi sama (5,0 %, terendah sasaran per kabupaten), 1.000 undian; tau peringkat provinsi
     lawan peringkat Kalimantan dihitung pada data yang disamakan prevalensinya.
  D. Selisih tau berpasangan. Per provinsi, tau(provinsi, Kalimantan) per kabupaten minus gabungan
     pada resampel kabupaten yang sama (1.000 ulangan, klaster kabupaten).
  E. Dekomposisi efek identitas provinsi (sasaran gabungan). Skor GB tanpa identitas digeser per
     provinsi pada skala logit supaya rerata logitnya sama dengan model + identitas (penyesuaian
     laju dasar saja, tanpa mengubah urutan di dalam provinsi). Bagian laju dasar = AUC-PR geser -
     AUC-PR asli; bagian dalam-provinsi = AUC-PR + identitas - AUC-PR geser. Ini dekomposisi
     diagnostik yang memakai skor uji, bukan model.

Pakai:    python kalimantan_keputusan.py DL_FIRE_NASIONAL/prediksi_kalimantan.csv
Keluaran: layar (simpan ke DL_FIRE_NASIONAL/oni_akhir/kalimantan_keputusan.log)
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score as ap

P = pd.read_csv(sys.argv[1])
P = P[P.skema == "rolling"].copy()
TUJUH = ["klimatologi", "persistence", "seasonal-naive", "rasio-analog", "regresi-ONI", "regresi-penuh",
         "gradient-boosting"]
NAMA = {"klimatologi": "climatology", "persistence": "persistence", "seasonal-naive": "seasonal naive",
        "rasio-analog": "ratio scaling", "regresi-ONI": "logistic ONI", "regresi-penuh": "logistic full",
        "gradient-boosting": "gradient boosting"}
TAFSIR = ["KalimantanBarat", "KalimantanTengah", "KalimantanSelatan", "KalimantanTimur"]
rng = np.random.default_rng(2026)
P["periode"] = P.tahun.astype(str) + "-" + P.bulan.astype(str)


def bagian_A():
    print("\nA. Anggaran peringatan tetap: bagian kabupaten-bulan parah yang tertangkap [95 %, bootstrap bulan]")
    for s in ("gabungan", "kabupaten"):
        d = P[P.sasaran == s]
        bulan = d.periode.unique()
        for k in (5, 10):
            teks = []
            for m in TUJUH:
                top = d.assign(r=d.groupby("periode")[m].rank(ascending=False, method="first"))
                hit = top[top.r <= k].groupby("periode").y.sum().reindex(bulan, fill_value=0)
                pos = d.groupby("periode").y.sum().reindex(bulan, fill_value=0)
                v = hit.sum() / pos.sum()
                bs = []
                for _ in range(2000):
                    i = rng.choice(len(bulan), len(bulan))
                    if pos.values[i].sum():
                        bs.append(hit.values[i].sum() / pos.values[i].sum())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                teks.append((v, f"{NAMA[m]} {v:.2f} [{lo:.2f}, {hi:.2f}]"))
            teks.sort(reverse=True)
            print(f"  {s:9s} k={k:2d} ({int(d.y.sum())} parah): " + "; ".join(t for _, t in teks))


def peringkat(d):
    return {m: ap(d.y, d[m]) for m in TUJUH}


def bagian_B():
    print("\nB. Penyesalan memilih model dari provinsi lain (AUC-PR model terbaik di P minus model pilihan)")
    for s in ("gabungan", "kabupaten"):
        d = P[P.sasaran == s]
        skor = {p: peringkat(d[d.provinsi == p]) for p in TAFSIR}
        kal = peringkat(d)
        terbaik_kal = max(kal, key=kal.get)
        sesal_q, sesal_k, salah = [], [], 0
        for p in TAFSIR:
            terbaik_p = max(skor[p].values())
            for q in TAFSIR:
                if q == p:
                    continue
                pilih = max(skor[q], key=skor[q].get)
                sesal_q.append(terbaik_p - skor[p][pilih])
                salah += pilih != max(skor[p], key=skor[p].get)
            sesal_k.append(terbaik_p - skor[p][terbaik_kal])
        print(f"  {s:9s}: dari provinsi lain rerata {np.mean(sesal_q):.3f} (maks {np.max(sesal_q):.3f}), "
              f"model pilihan bukan yang terbaik pada {salah} dari {len(sesal_q)} pasangan; "
              f"dari peringkat Kalimantan ({NAMA[terbaik_kal]}) rerata {np.mean(sesal_k):.3f} (maks {np.max(sesal_k):.3f})")


def bagian_C():
    print("\nC. Sasaran gabungan dengan prevalensi disamakan 5,0 % per provinsi (1.000 undian): tau lawan Kalimantan")
    d = P[P.sasaran == "gabungan"]
    kal = [ap(d.y, d[m]) for m in TUJUH]
    for p in TAFSIR:
        g = d[d.provinsi == p]
        pos, neg = g[g.y == 1], g[g.y == 0]
        n_neg = int(round(len(pos) * 0.95 / 0.05))
        if n_neg > len(neg):                       # prevalensi sudah di bawah 5 %: subsampel positif
            n_pos = int(round(len(neg) * 0.05 / 0.95))
            taus = [kendalltau([ap(x.y, x[m]) for m in TUJUH], kal)[0]
                    for x in (pd.concat([pos.sample(n_pos, random_state=int(rng.integers(1 << 30))), neg]) for _ in range(1000))]
        else:
            taus = [kendalltau([ap(x.y, x[m]) for m in TUJUH], kal)[0]
                    for x in (pd.concat([pos, neg.sample(n_neg, random_state=int(rng.integers(1 << 30)))]) for _ in range(1000))]
        asli = kendalltau([ap(g.y, g[m]) for m in TUJUH], kal)[0]
        print(f"  {p:18s} prevalensi asli {g.y.mean() * 100:4.1f}%  tau asli {asli:.3f}  "
              f"tau disamakan median {np.median(taus):.3f} [{np.percentile(taus, 2.5):.3f}, {np.percentile(taus, 97.5):.3f}]")


def bagian_D():
    print("\nD. Selisih tau berpasangan, per kabupaten - gabungan [95 %, klaster kabupaten, 1.000 ulangan]")
    kunci = ["provinsi", "kabupaten", "tahun", "bulan"]
    g = P[P.sasaran == "gabungan"].set_index(kunci)
    k = P[P.sasaran == "kabupaten"].set_index(kunci).reindex(g.index)
    for p in TAFSIR:
        idx_p = g.index.get_level_values("provinsi") == p
        kab = g.index.get_level_values("kabupaten")
        kab_p = np.unique(kab[idx_p])
        semua_kab = np.unique(kab)

        def tau(dg, dk, mask):
            out = []
            for dd in (dg, dk):
                kal = [ap(dd.y, dd[m]) for m in TUJUH]
                pr = [ap(dd.y[mask], dd[m][mask]) for m in TUJUH]
                out.append(kendalltau(pr, kal)[0])
            return out[1] - out[0]

        asli = tau(g, k, idx_p)
        bs = []
        for _ in range(1000):
            pilih = rng.choice(semua_kab, len(semua_kab))
            baris = np.concatenate([np.where(kab == x)[0] for x in pilih])
            dg, dk = g.iloc[baris], k.iloc[baris]
            mask = dg.index.get_level_values("provinsi") == p
            if dg.y[mask].sum() and dk.y[mask].sum():
                bs.append(tau(dg, dk, mask))
        print(f"  {p:18s} tau(kabupaten) - tau(gabungan) {asli:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")


def bagian_E():
    print("\nE. Dekomposisi efek identitas provinsi, sasaran gabungan (AUC-PR seluruh Kalimantan)")
    d = P[P.sasaran == "gabungan"].copy()
    eps = 1e-6
    lg = lambda x: np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps)))
    d["la"], d["lb"] = lg(d["gradient-boosting"]), lg(d["gb_prov"])
    geser = d.groupby("provinsi").lb.transform("mean") - d.groupby("provinsi").la.transform("mean")
    d["geser"] = 1 / (1 + np.exp(-(d.la + geser)))
    a, c, b = ap(d.y, d["gradient-boosting"]), ap(d.y, d["geser"]), ap(d.y, d["gb_prov"])
    print(f"  tanpa identitas {a:.3f}; geser laju dasar per provinsi {c:.3f}; + identitas {b:.3f}")
    print(f"  total {b - a:+.3f} = laju dasar {c - a:+.3f} + dalam-provinsi {b - c:+.3f}")
    for p in TAFSIR:
        g = d[d.provinsi == p]
        print(f"    {p:18s} dalam provinsi: tanpa {ap(g.y, g['gradient-boosting']):.3f}, + identitas {ap(g.y, g['gb_prov']):.3f}")


for f in (bagian_A, bagian_B, bagian_C, bagian_D, bagian_E):
    f()
