"""
PeatFireBench — nilai keputusan peringatan (telaah editor Natural Hazards, 22 Oktober 2026, X1).

Pertanyaannya: bila sebuah BPBD memakai prakiraan untuk memutuskan siaga atau tidak, berapa yang
didapat? Dua ukuran, keduanya dihitung per tahun uji lalu dirata-rata dalam kelompok tahun, pada
sasaran per kabupaten (kabupaten dengan paling sedikit satu bulan positif, seperti tabel naskah):

1. Aturan siaga tetap: tiap tahun, siaga pada 10 % kabupaten-bulan berskor tertinggi. Dilaporkan
   hit rate (POD, bagian bulan parah yang tersiaga) dan false alarm ratio (FAR, bagian siaga yang
   tidak parah). Aturan berbasis peringkat, jadi tidak bergantung pada kalibrasi.
2. Nilai ekonomi relatif (Richardson 2000) untuk rasio biaya/rugi a: V = (E_klim - E_prak) /
   (E_klim - E_sempurna), dengan E_klim = min(a, s), E_sempurna = a*s, E_prak = F*a*(1-s) + H*a*s +
   (1-H)*s, s = prevalensi tahun itu, H = hit rate, F = false alarm rate (POFD). Dua versi:
   (a) amplop: ambang terbaik untuk tiap a (batas atas, tanpa kalibrasi); (b) siaga bila peluang > a
   (keputusan optimal bila peluang terkalibrasi; hanya untuk model berpeluang). Selisih (a) - (b) =
   nilai yang hilang karena salah kalibrasi.

Kelompok tahun: biasa = S1 2020-2025 (enam tahun, dilatih dengan tahun sebelumnya saja); ekstrem =
S2 2014, 2015, 2019 (tahun itu disisihkan). Selang 95 %: bootstrap kabupaten, 500 ulangan, untuk
selisih nilai gradient boosting - klimatologi.

Pakai:    python patokan_nilai.py   (DL_FIRE_NASIONAL/penuh/patokan_nilai.log)
"""
import os

import numpy as np
import pandas as pd

from kelompok_fitur import PREDIKSI

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
P = pd.read_csv(os.path.join(D, PREDIKSI), dtype={"tag": str})
P = P[(P.sasaran == "y_kabupaten") & (P.ada_positif == 1)]
MODEL = ["klimatologi", "persistence", "fwi", "fwi_logistik", "logistik", "rf", "xgb", "mlp"]
PELUANG = ["fwi_logistik", "logistik", "rf", "xgb", "mlp"]
ALFA = [0.05, 0.1, 0.2, 0.3, 0.5]
KUOTA = 0.10
KELOMPOK = {"biasa (S1 2020-2025)": [("S1", str(t)) for t in range(2020, 2026)],
            "ekstrem (S2 2014, 2015, 2019)": [("S2", "2014"), ("S2", "2015"), ("S2", "2019")]}
rng = np.random.default_rng(2026)


def siaga_tetap(y, p):
    k = max(1, int(round(KUOTA * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    tp = y[top].sum()
    return tp / max(y.sum(), 1), 1 - tp / k


def nilai(H, F, s, a):
    e_klim, e_semp = min(a, s), a * s
    return (e_klim - (F * a * (1 - s) + H * a * s + (1 - H) * s)) / (e_klim - e_semp)


def kurva_roc(y, p):
    o = np.argsort(-p, kind="stable")
    ys, ps = y[o], p[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    ujung = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]  # titik ambang di antara skor berbeda
    npos, nneg = max(ys.sum(), 1), max((1 - ys).sum(), 1)
    return np.r_[0, tp[ujung] / npos], np.r_[0, fp[ujung] / nneg]


def amplop(y, p, a):
    H, F = kurva_roc(y, p)
    return nilai(H, F, y.mean(), a).max()


def pada_peluang(y, p, a):
    s, siaga = y.mean(), p > a
    H = (siaga & (y == 1)).sum() / max(y.sum(), 1)
    F = (siaga & (y == 0)).sum() / max((y == 0).sum(), 1)
    return nilai(H, F, s, a)


def baris(d):
    return d.y.to_numpy(), {m: d[m].to_numpy(dtype=float) for m in MODEL}


tahun = {k: {(sk, t): P[(P.skema == sk) & (P.tag == t)] for sk, t in v} for k, v in KELOMPOK.items()}

for kel, data in tahun.items():
    print(f"\n=== {kel}, sasaran per kabupaten")
    print(f"  prevalensi per tahun: " + ", ".join(f"{t} {d.y.mean() * 100:.1f}%" for (_, t), d in data.items()))
    print(f"\n  1) siaga 10 % kabupaten-bulan tertinggi tiap tahun: hit rate / false alarm ratio (rata-rata tahun)")
    for m in MODEL:
        r = np.array([siaga_tetap(*[d.y.to_numpy(), d[m].to_numpy(dtype=float)]) for d in data.values()])
        per = " ".join(f"{t}:{h:.2f}" for (_, t), (h, _) in zip(data, r))
        print(f"     {m:13s} hit {r[:, 0].mean():.3f}  FAR {r[:, 1].mean():.3f}   (hit per tahun {per})")
    print(f"\n  2a) nilai ekonomi relatif, amplop (ambang terbaik), rata-rata tahun, a = {ALFA}")
    for m in MODEL:
        v = [np.mean([amplop(d.y.to_numpy(), d[m].to_numpy(dtype=float), a) for d in data.values()]) for a in ALFA]
        print(f"     {m:13s} " + "  ".join(f"{x:+.3f}" for x in v))
    print(f"\n  2b) nilai bila siaga saat peluang > a (keputusan dengan peluang apa adanya)")
    for m in PELUANG:
        v = [np.mean([pada_peluang(d.y.to_numpy(), d[m].to_numpy(dtype=float), a) for d in data.values()]) for a in ALFA]
        g = [np.mean([amplop(d.y.to_numpy(), d[m].to_numpy(dtype=float), a) - pada_peluang(d.y.to_numpy(), d[m].to_numpy(dtype=float), a)
                      for d in data.values()]) for a in ALFA]
        print(f"     {m:13s} " + "  ".join(f"{x:+.3f}" for x in v) + "   hilang karena kalibrasi: " + "  ".join(f"{x:.3f}" for x in g))

    print(f"\n  3) selisih nilai amplop gradient boosting - klimatologi, selang bootstrap kabupaten (500)")
    for a in (0.1, 0.2):
        def stat(sub):
            return np.mean([amplop(dd.y.to_numpy(), dd.xgb.to_numpy(), a) - amplop(dd.y.to_numpy(), dd.klimatologi.to_numpy(dtype=float), a)
                            for dd in sub])
        titik = stat(data.values())
        daftar = list(data.values())
        gid = np.unique(daftar[0].gid)
        letak = [{g: np.where(d.gid.to_numpy() == g)[0] for g in gid} for d in daftar]
        boot = []
        for _ in range(500):
            pilih = rng.choice(gid, len(gid))
            boot.append(stat([d.iloc[np.concatenate([lt[g] for g in pilih])] for d, lt in zip(daftar, letak)]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"     a = {a}: {titik:+.3f} [{lo:+.3f}, {hi:+.3f}]")
