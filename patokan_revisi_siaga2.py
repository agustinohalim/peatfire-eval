"""
PeatFireBench revisi fase D7 (Hasil_Patokan_9): kuota 10 % per bulan lawan ambang luar-lipatan, pada
pijakan sama (hit rate, false alarm ratio, bagian siaga, nilai ekonomi relatif), tahun biasa dan ekstrem,
per kabupaten. Ambang: (i) kuantil 0,9 prakiraan luar-lipatan tahun latih (aturan a fase C4) dan
(ii) ambang yang memaksimalkan nilai rata-rata per tahun dalam luar-lipatan untuk tiap a (seperti C4).
Fungsi disalin dari patokan_revisi_siaga.py (fase C4) tanpa perubahan.

Pakai:    python -u patokan_revisi_siaga2.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_siaga2.log
"""
import numpy as np
import pandas as pd

import patokan_revisi_bantu as B

ALFA = [0.05, 0.1, 0.2, 0.3, 0.5]
P = B.muat_P(fwi=False)
O = pd.read_csv(f"{B.D}/patokan_revisi_oof.csv", dtype={"tag": str})
O = O[O.ada_positif == 1]
KEL = {"biasa": [("S1", t) for t in B.BIASA], "ekstrem": [("S2", t) for t in B.EKSTREM]}


class BlokB(B.Blok):
    def __init__(self, g, kolom, kab):
        super().__init__(g, kolom, kab)
        self.bulan = g.bulan.to_numpy()


def kuota(b, j, m, frac=0.1):
    jj = np.arange(len(b.y))[j]
    bl = b.bulan[jj]; s = b.s[m][jj]; siaga = np.zeros(len(jj), bool)
    for mo in np.unique(bl):
        w = np.where(bl == mo)[0]; n = max(1, int(round(frac * len(w))))
        siaga[w[np.argsort(-s[w], kind="stable")[:n]]] = True
    return siaga


def nilai(H, F, s, a):
    e_klim, e_semp = min(a, s), a * s
    return (e_klim - (F * a * (1 - s) + H * a * s + (1 - H) * s)) / (e_klim - e_semp)


def ukur(y, siaga):
    tp = (siaga & (y == 1)).sum()
    H = tp / max(y.sum(), 1); F = (siaga & (y == 0)).sum() / max((y == 0).sum(), 1)
    return [H, 1 - tp / max(siaga.sum(), 1), siaga.mean()] + [nilai(H, F, y.mean(), a) for a in ALFA]


def ambang_oof(o, m, a):
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


B.cek_ap()
Pt = P[(P.sasaran == "y_kabupaten") & (P.ada_positif == 1)]
MK = ["xgb", "rf", "klim_frek1"]
MO = ["xgb", "klim_frek1"]
UK = ["hit", "FAR", "siaga"] + [f"REV{a}" for a in ALFA]
print("=" * 100 + "\nD7. Kuota lawan ambang luar-lipatan, per kabupaten\n" + "=" * 100)
for kel, kunci in KEL.items():
    kab = Pt.gid.unique()
    blok = {k: BlokB(Pt[(Pt.skema == k[0]) & (Pt.tag == k[1])], MK, kab) for k in kunci}
    oof = {k: O[(O.skema == k[0]) & (O.tag == k[1])] for k in kunci}
    thr10 = {(k, m): np.quantile(oof[k][m].to_numpy(dtype=float), 0.9) for k in kunci for m in MO}
    thrv = {(k, m, a): ambang_oof(oof[k], m, a) for k in kunci for m in MO for a in ALFA}
    for k, b in blok.items():
        b.kunci = k

    def fn(b, j):
        y = b.y[j]; out = []
        for m in MK:
            out += ukur(y, kuota(b, j, m))
        for m in MO:
            out += ukur(y, b.s[m][j] >= thr10[(b.kunci, m)])
        for m in MO:
            out += [nilai(*(lambda s: ((s & (y == 1)).sum() / max(y.sum(), 1), (s & (y == 0)).sum() / max((y == 0).sum(), 1)))(
                b.s[m][j] >= thrv[(b.kunci, m, a)]), y.mean(), a) for a in ALFA]
        return np.array(out, dtype=float)

    nama = ([f"kuota {m} {u}" for m in MK for u in UK] + [f"amb10 {m} {u}" for m in MO for u in UK]
            + [f"ambnilai {m} REV{a}" for m in MO for a in ALFA])
    ix = {n: i for i, n in enumerate(nama)}
    th = np.array([fn(blok[k], slice(None)) for k in kunci])
    titik = th.mean(0)
    print(f"\n--- {kel}: prevalensi {np.mean([blok[k].y.mean() for k in kunci]) * 100:.2f}%")
    print(f"    {'aturan':<24}{'hit':>7}{'FAR':>7}{'siaga':>8}   REV a = " + " / ".join(str(a) for a in ALFA))
    for aturan, ms in (("kuota", MK), ("amb10", MO)):
        for m in ms:
            print(f"    {aturan + ' ' + m:<24}{titik[ix[f'{aturan} {m} hit']]:>7.3f}{titik[ix[f'{aturan} {m} FAR']]:>7.3f}"
                  f"{titik[ix[f'{aturan} {m} siaga']] * 100:>7.1f}%   "
                  + " / ".join(f"{titik[ix[f'{aturan} {m} REV{a}']]:+.3f}" for a in ALFA))
    for m in MO:
        print(f"    {'ambnilai ' + m:<24}{'':>22}   " + " / ".join(f"{titik[ix[f'ambnilai {m} REV{a}']]:+.3f}" for a in ALFA))
    print("    per tahun hit kuota xgb / amb10 xgb / kuota EF-1m / amb10 EF-1m: " + "  ".join(
        f"{k[1]} {x[ix['kuota xgb hit']]:.3f}/{x[ix['amb10 xgb hit']]:.3f}/{x[ix['kuota klim_frek1 hit']]:.3f}/"
        f"{x[ix['amb10 klim_frek1 hit']]:.3f}" for k, x in zip(kunci, th)))
    if kel == "biasa":
        R = B.dua_tahap(blok, fn, 1000, np.random.default_rng(B.SEED), kab).mean(1)
        for m in MO:
            for a in ALFA:
                for lain in ("amb10", "ambnilai"):
                    i, r = ix[f"kuota {m} REV{a}"], ix[f"{lain} {m} REV{a}"]
                    lo, hi = np.nanpercentile(R[:, i] - R[:, r], [2.5, 97.5])
                    print(f"    REV kuota - {lain} {m:<11} a={a:<5}{titik[i] - titik[r]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    else:
        for m in MO:
            for a in ALFA:
                print(f"    REV kuota - amb10 {m} a={a}: " + "  ".join(
                    f"{k[1]} {x[ix[f'kuota {m} REV{a}']] - x[ix[f'amb10 {m} REV{a}']]:+.3f}" for k, x in zip(kunci, th)))
print("\nselesai.")
