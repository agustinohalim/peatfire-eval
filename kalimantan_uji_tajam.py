"""
Naskah gabungan Kalimantan — uji keberatan panel telaah kedua (Telaah/2026-10-01b).

Apakah temuan utama (memilih model dari provinsi lain salah dengan ambang gabungan, tidak pernah
dengan ambang per kabupaten) bertahan bila:

  M1  cakupan latih: gradient boosting diganti versi yang dilatih pada provinsinya sendiri (gb_lokal);
  X1  baseline adil untuk ambang per kabupaten: skor baseline lokal dibagi ambang kabupatennya,
      dan klimatologi diganti frekuensi lampau melampaui ambang (bulan-kalender, tahun latih);
  X2  ketidakpastian: seluruh prosedur (juara per provinsi lalu penyesalan) diulang pada 1.000
      resampel kabupaten di dalam tiap provinsi; dilaporkan selang penyesalan rerata, selang selisih
      gabungan - per kabupaten, dan peluang juara berbeda;
  M3  bias seleksi: juara dipilih pada 2019-2022 lalu penyesalan diukur pada 2023-2025;
  P1  satuan bahaya: di provinsi P, tiap bulan ditandai 20 % kabupaten berskor tertinggi; dihitung
      kabupaten-bulan parah yang hilang bila memakai model juara provinsi lain dibanding juara P.

Pakai:    python kalimantan_uji_tajam.py DL_FIRE_NASIONAL/panel_kalimantan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt
Keluaran: layar (simpan ke DL_FIRE_NASIONAL/oni_akhir/kalimantan_uji_tajam.log)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py"), encoding="utf-8").read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
df, _ = muat(panel_path, oni_path)
mentah = pd.read_csv(panel_path)
mentah = mentah[mentah["tahun"] <= 2025]
q_kab = mentah.groupby("kabupaten")["titik_panas"].quantile(0.90)
P = pd.read_csv(os.path.join(os.path.dirname(panel_path), "prediksi_kalimantan.csv"))
P = P[P.skema == "rolling"].copy()
P["bulan_ke"] = P.bulan.str[-2:].astype(int) if P.bulan.dtype == object else P.bulan

MODEL = ["klimatologi", "persistence", "seasonal-naive", "rasio-analog", "regresi-ONI", "regresi-penuh",
         "gradient-boosting"]
LOKAL = ["klimatologi", "persistence", "seasonal-naive", "rasio-analog"]
TAFSIR = ["KalimantanBarat", "KalimantanTengah", "KalimantanSelatan", "KalimantanTimur"]
rng = np.random.default_rng(2026)

# --- baseline adil untuk sasaran per kabupaten
thr = P.kabupaten.map(q_kab).clip(lower=1.0)
d_kab = df.copy()
d_kab["lewat"] = (d_kab["titik_panas"] > d_kab["kabupaten"].map(q_kab)).astype(float)
frek = []
for th in sorted(P.tahun.unique()):
    lalu = d_kab[d_kab.tahun < th].groupby(["kabupaten", "bulan_ke"]).lewat.mean()
    blok = P[P.tahun == th][["kabupaten", "bulan_ke"]]
    frek.append(pd.Series(lalu.reindex(pd.MultiIndex.from_frame(blok)).fillna(0).to_numpy(), index=blok.index))
P["frek_lampau"] = pd.concat(frek).reindex(P.index)
for m in LOKAL:
    P[m + "_adil"] = P[m] / thr
ADIL = ["frek_lampau", "persistence_adil", "seasonal-naive_adil", "rasio-analog_adil",
        "regresi-ONI", "regresi-penuh", "gradient-boosting"]


def skor(d, model):
    return {m: ap(d.y, d[m]) for m in model} if d.y.sum() else None


def sesal(d, model):
    """Juara per provinsi lalu penyesalan rerata atas 12 pasangan; juga jumlah pasangan salah."""
    s = {p: skor(d[d.provinsi == p], model) for p in TAFSIR}
    if any(v is None for v in s.values()):
        return None
    juara = {p: max(s[p], key=s[p].get) for p in TAFSIR}
    r = [s[p][juara[p]] - s[p][juara[q]] for p in TAFSIR for q in TAFSIR if q != p]
    salah = sum(juara[p] != juara[q] for p in TAFSIR for q in TAFSIR if q != p)
    return np.mean(r), salah, juara


def boot(d, model, n=1000):
    """Resampel kabupaten di dalam tiap provinsi, lalu seluruh prosedur diulang."""
    idx = {p: {k: np.where((d.provinsi == p) & (d.kabupaten == k))[0] for k in d[d.provinsi == p].kabupaten.unique()}
           for p in TAFSIR}
    hasil = []
    for _ in range(n):
        baris = np.concatenate([idx[p][k] for p in TAFSIR for k in rng.choice(list(idx[p]), len(idx[p]))])
        h = sesal(d.iloc[baris], model)
        if h:
            hasil.append(h[:2])
    return np.array(hasil)


def nama(j):
    return {p[10:]: v.replace("gradient-boosting", "GB").replace("_adil", "*") for p, v in j.items()}


rakit = {
    "gabungan": [("asli", MODEL)],
    "kabupaten": [("asli", MODEL), ("baseline adil", ADIL)],
}
print("Penyesalan memilih juara provinsi lain (AUC-PR), rerata 12 pasangan [95 %, 1.000 resampel kabupaten]")
simpan = {}
for s, varian in rakit.items():
    d = P[P.sasaran == s]
    for label, model in varian:
        for cakupan in ("seluruh Kalimantan", "GB dilatih per provinsi"):
            dd = d.copy()
            if cakupan.startswith("GB dilatih"):
                dd["gradient-boosting"] = dd["gb_lokal"].fillna(dd["gradient-boosting"])
            r, salah, juara = sesal(dd, model)
            b = boot(dd, model)
            simpan[(s, label, cakupan)] = b
            print(f"  {s:9s} {label:13s} {cakupan:24s} sesal {r:.3f} [{np.percentile(b[:, 0], 2.5):.3f}, "
                  f"{np.percentile(b[:, 0], 97.5):.3f}]  pasangan salah {salah}/12 "
                  f"(resampel: median {int(np.median(b[:, 1]))}, P(0 salah) {np.mean(b[:, 1] == 0):.2f})  juara {nama(juara)}")

print("\nSelisih penyesalan gabungan - per kabupaten (berpasangan per resampel tidak tersedia antar-sasaran;"
      " dihitung dari resampel independen)")
for cak in ("seluruh Kalimantan", "GB dilatih per provinsi"):
    for lab in ("asli", "baseline adil"):
        a, b = simpan[("gabungan", "asli", cak)][:, 0], simpan[("kabupaten", lab, cak)][:, 0]
        n = min(len(a), len(b))
        dif = a[:n] - b[:n]
        print(f"  {cak:24s} per kabupaten {lab:13s}: {np.mean(a) - np.mean(b):+.3f} "
              f"[{np.percentile(dif, 2.5):+.3f}, {np.percentile(dif, 97.5):+.3f}]")

print("\nM3: juara dipilih pada 2019-2022, penyesalan diukur pada 2023-2025")
for s, varian in rakit.items():
    for label, model in varian:
        d = P[P.sasaran == s]
        pilih, uji = d[d.tahun <= 2022], d[d.tahun >= 2023]
        sp = {p: skor(pilih[pilih.provinsi == p], model) for p in TAFSIR}
        su = {p: skor(uji[uji.provinsi == p], model) for p in TAFSIR}
        if any(v is None for v in list(sp.values()) + list(su.values())):
            print(f"  {s} {label}: ada provinsi tanpa positif"); continue
        juara = {p: max(sp[p], key=sp[p].get) for p in TAFSIR}
        r = [max(su[p].values()) - su[p][juara[q]] for p in TAFSIR for q in TAFSIR if q != p]
        r_sendiri = [max(su[p].values()) - su[p][juara[p]] for p in TAFSIR]
        print(f"  {s:9s} {label:13s} sesal pakai juara provinsi lain {np.mean(r):.3f}; pakai juara sendiri {np.mean(r_sendiri):.3f}")

print("\nP1: kabupaten-bulan parah yang hilang (anggaran 20 % kabupaten per bulan di tiap provinsi, 2019-2025)")
for s, varian in rakit.items():
    for label, model in varian:
        d = P[P.sasaran == s]
        _, _, juara = sesal(d, model)
        tot = []
        for p in TAFSIR:
            g = d[d.provinsi == p]
            k = max(1, int(round(0.2 * g.kabupaten.nunique())))
            def tangkap(m):
                r = g.groupby(["tahun", "bulan"])[m].rank(ascending=False, method="first")
                return int(g.y[r <= k].sum())
            for q in TAFSIR:
                if q != p:
                    tot.append((tangkap(juara[p]) - tangkap(juara[q]), int(g.y.sum())))
        hilang = np.array(tot)
        print(f"  {s:9s} {label:13s} rerata {hilang[:, 0].mean():.1f} kabupaten-bulan parah hilang per pasangan "
              f"(maks {hilang[:, 0].max()}, dari {hilang[:, 1].mean():.0f} parah per provinsi); "
              f"{np.mean(hilang[:, 0] > 0) * 100:.0f}% pasangan kehilangan")
