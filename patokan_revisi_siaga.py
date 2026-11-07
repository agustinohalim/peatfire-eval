"""
PeatFireBench revisi fase C4 (Hasil_Patokan_8): aturan siaga dan nilai keputusan terhadap EF-1m
(klim_frek1), dengan selang kabupaten dan dua tahap.

Pakai:    python -u patokan_revisi_siaga.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_siaga.log
"""
import numpy as np
import pandas as pd

import patokan_revisi_bantu as B

ALFA = [0.05, 0.1, 0.2, 0.3, 0.5]
P = B.muat_P(fwi=False)
O = pd.read_csv(f"{B.D}/patokan_revisi_oof.csv", dtype={"tag": str})
O = O[O.ada_positif == 1]
KEL = {"biasa (S1 2020-2025)": [("S1", t) for t in B.BIASA], "ekstrem (S2 2015, 2014, 2019)": [("S2", t) for t in B.EKSTREM]}


class BlokB(B.Blok):
    def __init__(self, g, kolom, kab):
        super().__init__(g, kolom, kab)
        self.bulan = g.bulan.to_numpy()


def kuota(b, j, m, frac=0.1):
    """indeks (dalam j) yang disiagakan: 10 % baris tertinggi tiap bulan."""
    jj = np.arange(len(b.y))[j]
    bl = b.bulan[jj]; s = b.s[m][jj]; siaga = np.zeros(len(jj), bool)
    for mo in np.unique(bl):
        w = np.where(bl == mo)[0]; n = max(1, int(round(frac * len(w))))
        siaga[w[np.argsort(-s[w], kind="stable")[:n]]] = True
    return siaga


def hit_far(y, siaga):
    tp = (siaga & (y == 1)).sum()
    return tp / max(y.sum(), 1), 1 - tp / max(siaga.sum(), 1)


def nilai(H, F, s, a):
    e_klim, e_semp = min(a, s), a * s
    return (e_klim - (F * a * (1 - s) + H * a * s + (1 - H) * s)) / (e_klim - e_semp)


def nilai_siaga(y, siaga, a):
    H = (siaga & (y == 1)).sum() / max(y.sum(), 1); F = (siaga & (y == 0)).sum() / max((y == 0).sum(), 1)
    return nilai(H, F, y.mean(), a)


def amplop(y, p, a):
    o = np.argsort(-p, kind="stable"); ys, ps = y[o], p[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    u = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]
    H = np.r_[0, tp[u] / max(ys.sum(), 1)]; F = np.r_[0, fp[u] / max((1 - ys).sum(), 1)]
    return nilai(H, F, y.mean(), a).max()


def ambang_oof(o, m, a):
    """ambang yang memaksimalkan rata-rata nilai per tahun dalam pada prakiraan luar-lipatan (seperti A4)."""
    calon = np.unique(np.quantile(o[m].to_numpy(dtype=float), np.linspace(0, 1, 401)))
    tot = np.zeros(len(calon)); n = 0
    for _, g in o.groupby("tahun_dalam"):
        y = g.y.to_numpy(); p = g[m].to_numpy(dtype=float)
        if y.sum() == 0:
            continue
        pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
        H = 1 - np.searchsorted(pos, calon, side="left") / len(pos)
        F = 1 - np.searchsorted(neg, calon, side="left") / len(neg)
        tot += nilai(H, F, y.mean(), a); n += 1
    return calon[int(np.argmax(tot / n))]


def cetak_selang(nama, titik, kb, dt):
    print(f"    {nama:<34}{B.fmt(titik, *B.ci(kb))}  dua tahap [{B.ci(dt)[0]:+.3f}, {B.ci(dt)[1]:+.3f}]")


B.cek_ap()
print("=" * 100 + "\nC4. Kuota bulanan, ambang luar-lipatan, nilai ekonomi relatif, per kabupaten\n" + "=" * 100)
Pt = P[(P.sasaran == "y_kabupaten") & (P.ada_positif == 1)]
MK = ["xgb", "rf", "mlp", "log_klim_jeda", "klim_frek1", "klim_frek", "klimatologi"]
MO = ["xgb", "klim_frek1", "klim_frek", "klimatologi", "log_klim_jeda"]
MP = ["xgb", "klim_frek1", "klim_frek"]
for kel, kunci in KEL.items():
    uji = {k: Pt[(Pt.skema == k[0]) & (Pt.tag == k[1])] for k in kunci}
    oof = {k: O[(O.skema == k[0]) & (O.tag == k[1])] for k in kunci}
    kab = Pt.gid.unique()
    blok = {k: BlokB(uji[k], MK, kab) for k in kunci}
    thr10 = {(k, m): np.quantile(oof[k][m].to_numpy(dtype=float), 0.9) for k in kunci for m in MO}
    thrv = {(k, m, a): ambang_oof(oof[k], m, a) for k in kunci for m in MO for a in ALFA}

    def fn(b, j, k=None):
        y = b.y[j]; out = []
        for m in MK:
            out += list(hit_far(y, kuota(b, j, m)))
        for m in MO:
            out += list(hit_far(y, b.s[m][j] >= thr10[(b.kunci, m)])) + [(b.s[m][j] >= thr10[(b.kunci, m)]).mean()]
        for m in MP:
            out += [nilai_siaga(y, b.s[m][j] > a, a) for a in ALFA]
        for m in MO:
            out += [nilai_siaga(y, b.s[m][j] >= thrv[(b.kunci, m, a)], a) for a in ALFA]
        return np.array(out, dtype=float)

    for k, b in blok.items():
        b.kunci = k
    nama = ([f"kuota {m} {u}" for m in MK for u in ("hit", "FAR")] + [f"oof10 {m} {u}" for m in MO for u in ("hit", "FAR", "siaga")]
            + [f"p>a {m} {a}" for m in MP for a in ALFA] + [f"thr {m} {a}" for m in MO for a in ALFA])
    ix = {n: i for i, n in enumerate(nama)}
    titik = np.mean([fn(b, slice(None)) for b in blok.values()], axis=0)
    rng = np.random.default_rng(B.SEED)
    kb = B.ulang(blok, fn, 500, False, rng, kab); dt = B.ulang(blok, fn, 1000, True, rng, kab)
    print(f"\n--- {kel}")
    print("  (b) kuota 10 % kabupaten tertinggi tiap bulan: hit / FAR [hit kab] [hit dua tahap]")
    for m in MK:
        i, f = ix[f"kuota {m} hit"], ix[f"kuota {m} FAR"]
        print(f"    {m:<14} hit {titik[i]:.3f} [{B.ci(kb[:, i])[0]:.3f}, {B.ci(kb[:, i])[1]:.3f}] "
              f"[{B.ci(dt[:, i])[0]:.3f}, {B.ci(dt[:, i])[1]:.3f}]  FAR {titik[f]:.3f}", flush=True)
    for m in ("xgb", "rf", "mlp", "log_klim_jeda"):
        i, r = ix[f"kuota {m} hit"], ix["kuota klim_frek1 hit"]
        cetak_selang(f"selisih hit {m} - EF-1m", titik[i] - titik[r], kb[:, i] - kb[:, r], dt[:, i] - dt[:, r])
    print("    per tahun hit xgb / EF-1m: " + "  ".join(
        f"{k[1]} {fn(blok[k], slice(None))[ix['kuota xgb hit']]:.3f}/{fn(blok[k], slice(None))[ix['kuota klim_frek1 hit']]:.3f}"
        for k in kunci))
    print("  (a) ambang 10 % luar-lipatan: hit / FAR / bagian siaga")
    for m in MO:
        print(f"    {m:<14} hit {titik[ix[f'oof10 {m} hit']]:.3f}  FAR {titik[ix[f'oof10 {m} FAR']]:.3f}  "
              f"siaga {titik[ix[f'oof10 {m} siaga']] * 100:.1f}%")
    print("  nilai ekonomi relatif, a = " + " / ".join(str(a) for a in ALFA))
    for m in MP:
        print(f"    p>a   {m:<12}" + " ".join(f"{titik[ix[f'p>a {m} {a}']]:+.3f}" for a in ALFA))
    for m in MO:
        print(f"    thr   {m:<12}" + " ".join(f"{titik[ix[f'thr {m} {a}']]:+.3f}" for a in ALFA))
    for m in MP:
        print(f"    amplop {m:<11}" + " ".join(
            f"{np.mean([amplop(uji[k].y.to_numpy(), uji[k][m].to_numpy(dtype=float), a) for k in kunci]):+.3f}" for a in ALFA))
    for a in ALFA:
        for jenis in ("p>a", "thr"):
            i, r = ix[f"{jenis} xgb {a}"], ix[f"{jenis} klim_frek1 {a}"]
            cetak_selang(f"{jenis} xgb - EF-1m, a = {a}", titik[i] - titik[r], kb[:, i] - kb[:, r], dt[:, i] - dt[:, r])

print("\n" + "=" * 100 + "\nC4 nasional: kuota bulanan, 498 kabupaten\n" + "=" * 100)
Pn = P[P.sasaran == "y_gabungan"]
MN = ["xgb", "rf", "log_klim_jeda", "klim_frek", "klim_frek1", "klimatologi"]
for kel, kunci in KEL.items():
    kab = Pn.gid.unique()
    blok = {k: BlokB(Pn[(Pn.skema == k[0]) & (Pn.tag == k[1])], MN, kab) for k in kunci}

    def fn(b, j):
        y = b.y[j]; out = []
        for m in MN:
            out += list(hit_far(y, kuota(b, j, m)))
        return np.array(out)

    titik = np.mean([fn(b, slice(None)) for b in blok.values()], axis=0)
    rng = np.random.default_rng(B.SEED)
    kb = B.ulang(blok, fn, 500, False, rng, kab); dt = B.ulang(blok, fn, 1000, True, rng, kab)
    print(f"\n--- {kel}")
    for i, m in enumerate(MN):
        print(f"    {m:<14} hit {titik[2 * i]:.3f}  FAR {titik[2 * i + 1]:.3f}")
    r = 2 * MN.index("klim_frek")
    for m in ("xgb", "rf"):
        i = 2 * MN.index(m)
        cetak_selang(f"selisih hit {m} - EF-3m", titik[i] - titik[r], kb[:, i] - kb[:, r], dt[:, i] - dt[:, r])
print("\nselesai.")
