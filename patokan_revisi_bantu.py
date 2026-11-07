"""
PeatFireBench revisi fase C — fungsi bersama (Hasil_Patokan_8).

ap_cepat   AUC-PR (average precision) setara sklearn.average_precision_score, dengan numpy saja.
muat_P     prediksi S1/S2 dari patokan_revisi_prediksi.csv + fwi_persentil, klim_fwi (patokan_fwi_adil.csv).
Blok       satu blok tahun: y dan skor, dengan indeks baris per kabupaten untuk bootstrap.
ulang      selang kabupaten (tahun tetap) dan dua tahap (tahun lalu kabupaten) untuk statistik rata-rata tahun.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
SEED = 2026
BIASA = [str(t) for t in range(2020, 2026)]
EKSTREM = ["2015", "2014", "2019"]
TENANG = ["2020", "2021", "2022", "2024", "2025"]
PASANGAN = {"y_kabupaten": "klim_frek1", "y_gabungan": "klim_frek"}
LABEL = {"klim_frek1": "EF-1m", "klim_frek": "EF-3m", "klimatologi": "MC"}


def ap_cepat(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort"); y = y[o]; s = s[o]
    tp = np.cumsum(y)
    ujung = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tps = tp[ujung]; prec = tps / (ujung + 1); rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def cek_ap(rng=np.random.default_rng(0)):
    for _ in range(20):
        y = rng.integers(0, 2, 500); s = np.round(rng.random(500), 2)
        assert abs(ap_cepat(y, s) - average_precision_score(y, s)) < 1e-12
    print("cek ap_cepat = sklearn: ok", flush=True)


def muat_P(fwi=True):
    P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
    P = P[P.skema.isin(["S1", "S2"])]
    if fwi:
        kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
        F = pd.read_csv(os.path.join(D, "patokan_fwi_adil.csv"), dtype={"tag": str})
        P = P.merge(F[kunci + ["fwi_persentil", "klim_fwi"]], on=kunci, how="left", validate="one_to_one")
    return P


def ambil_sasaran(P, sas):
    d = P[P.sasaran == sas]
    return d[d.ada_positif == 1] if sas == "y_kabupaten" else d


class Blok:
    def __init__(self, g, kolom, kab):
        self.y = g.y.to_numpy()
        self.s = {m: g[m].to_numpy(dtype=float) for m in kolom}
        gid = g.gid.to_numpy()
        self.idx = {k: np.where(gid == k)[0] for k in kab}

    def baris(self, pilih):
        if pilih is None:
            return slice(None)
        return np.concatenate([self.idx[k] for k in pilih])


def ulang(blok, fn, n, dua_tahap, rng, kab):
    """blok: dict tag -> Blok. fn(Blok, j) -> vektor statistik satu tahun. Rata-rata atas tahun."""
    tags = list(blok)
    out = []
    for _ in range(n):
        pilih = rng.choice(kab, len(kab))
        tt = [tags[i] for i in rng.integers(0, len(tags), len(tags))] if dua_tahap else tags
        cache = {}
        for t in set(tt):
            cache[t] = fn(blok[t], blok[t].baris(pilih))
        out.append(np.nanmean([cache[t] for t in tt], axis=0))
    return np.array(out)


def fmt(x, lo, hi):
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def ci(a, axis=0):
    return np.nanpercentile(a, [2.5, 97.5], axis=axis)


def dua_tahap(blok, fn, n, rng, kab):
    """fase D: (n, banyak tahun, k) statistik per tahun terambil; tahun lalu kabupaten (tarikan kabupaten
    sama untuk semua tahun dalam satu ulangan). Rata-rata atas sumbu 1 = statistik `ulang(..., True)`."""
    tags = list(blok)
    out = None
    for r in range(n):
        pilih = rng.choice(kab, len(kab))
        tt = [tags[i] for i in rng.integers(0, len(tags), len(tags))]
        cache = {t: fn(blok[t], blok[t].baris(pilih)) for t in set(tt)}
        v = np.array([cache[t] for t in tt])
        if out is None:
            out = np.empty((n,) + v.shape)
        out[r] = v
    return out


def kab_tetap(b, fn, n, rng, kab):
    """fase D: selang kabupaten untuk satu tahun (tahun tetap)."""
    return np.array([fn(b, b.baris(rng.choice(kab, len(kab)))) for _ in range(n)])
