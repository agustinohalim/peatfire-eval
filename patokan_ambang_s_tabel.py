"""
Revisi fase S (Hasil_Patokan_25): Tabel 2, Tabel 4 dan Tabel 6 (kolom tahun biasa) dengan ambang dari tahun latih,
S1 2020-2025, dari prakiraan patokan_ambang_s.py. Satu sasaran per jalan.

(C3) Pilihan rujukan di dalam lipatan seperti patokan_revisi_rujukan.py (AUC-PR dalam-tahun luar-lipatan pada tahun
     latih, calon MC, EF-3m, EF-1m), sekarang pada label ulang; dicetak untuk aturan R10. Tabel 2 tetap memakai rujukan
     pasangan B.PASANGAN (EF-1m per kabupaten, EF-3m nasional).
(S1) Tabel 2: AUC-PR rata-rata tahun, keunggulan dalam-tahun atas rujukan pasangan, selang kabupaten dan dua tahap
     (1.000 ulangan, benih B.SEED; urutan tarikan seperti patokan_revisi_tahun.py: kabupaten dulu, lalu dua tahap),
     selang tahun t (df 5), banyak tahun > 0, per tahun, dan AUC-PR "pooled" (enam tahun sebagai satu himpunan).
(S2) Tabel 4: BSS dalam-tahun rata-rata atas kalender pasangan, EF-P, EF-L dan EF-LD pada sel yang sama dengan Tabel 4
     naskah; matriks S dan dua tahap cepat disalin dari patokan_tingkat_o.py:183-216; dua tahap benih B.SEED, 1, 2, 3, 4
     (1.000 ulangan masing-masing). Tanda: tebal = dua tahap (benih B.SEED) dan tahun t tanpa nol; † = dua tahap tanpa
     nol, tahun t dengan nol; ‡ = dua tahap tanpa nol pada sebagian dari lima benih, tahun t dengan nol.
(S3) Tabel 6 kolom tahun biasa (per kabupaten saja): nilai ekonomi relatif enam aturan siaga, a = 0,05/0,1/0,2/0,3/0,5;
     fungsi dan aturan disalin dari patokan_revisi_tingkat.py:243-395 tanpa perubahan isi (ambang dan kuota disetel
     pada baris tahun latih berlabel ulang dari patokan_ambang_s.py). Kontras R30, R31 dan kuota - ambang
     (patokan_revisi_siaga2.py:105-112) dengan dua tahap 1.000 benih B.SEED dan selang tahun t.
(Cek) BSS RF, GB dan EF-L atas kalender, format G2 (patokan_revisi_ambang_peluang.py:156-161), untuk dibandingkan
     dengan G2 (Hasil_Patokan_12) dan P (Hasil_Patokan_22).

Pakai:
    python -u patokan_ambang_s_tabel.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_ambang_s_tabel_kab.log
    python -u patokan_ambang_s_tabel.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_ambang_s_tabel_nas.log
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
REF = B.PASANGAN[SAS]
N = 1000
ALFA = [0.05, 0.1, 0.2, 0.3, 0.5]
BENIH = [B.SEED, 1, 2, 3, 4]
U = pd.read_csv(f"{B.D}/patokan_ambang_s_{SAS}.csv", dtype={"tag": str})
L = pd.read_csv(f"{B.D}/patokan_ambang_s_{SAS}_latih.csv", dtype={"tag": str})
assert sorted(U.tag.unique()) == B.BIASA
B.cek_ap()
NAMA = {"klimatologi": "MC", "klim_frek": "EF-3m", "klim_frek1": "EF-1m", "log_klim_jeda": "Local history, MC",
        "log_ef1_jeda": "Local history, EF-1m", "persistence": "Persistence", "seasonal_naive": "Seasonal naive",
        "rasio": "Ratio scaling", "fwi": "FWI, national scale", "fwi_persentil": "FWI, district percentile",
        "fwi_logistik": "Logistic (FWI)", "klim_fwi": "Climatology + FWI", "dc_nasional": "DC, national scale",
        "dc_persentil": "DC, district percentile", "ffmc_nasional": "FFMC, national scale",
        "log_ef1_fdrs": "Local history, EF-1m + DC/FFMC", "logistik": "Logistic (ONI, DMI)",
        "xgb_sejarah": "GB, fire history", "rf": "Random forest", "xgb": "Gradient boosting",
        "xgb_setel": "Gradient boosting, tuned", "mlp": "MLP", "konstan": "Constant rate", "rujukan_peluang": "EF-P",
        "ef1_tingkat": "EF-L", "efd": "EF-D", "efld": "EF-LD", "xgb_nas": "Gradient boosting + national count",
        "xgb_nasdc": "Gradient boosting + national count + DC"}


# ---- disalin dari patokan_revisi_setel.py:195-197 tanpa perubahan isi
def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se
# ---- akhir salinan


def tanpa_nol(lo, hi):
    return lo > 0 or hi < 0


print(f"{SAS}: rujukan pasangan {B.LABEL[REF]}; {U.gid.nunique()} kabupaten, {len(U):,} baris, {int(U.y.sum()):,} positif, "
      f"prevalensi {U.y.mean() * 100:.2f}%  (ambang tahun latih; per kabupaten: kabupaten positif di tahun latih lipatan)")
print("  per lipatan: " + "  ".join(f"{t} {x.gid.nunique()} kab {x.y.mean() * 100:.2f}%" for t, x in U.groupby("tag")))
print("  EF-P terpilih per lipatan: " + "  ".join(f"{t} {x.rujukan_peluang_calon.iloc[0]}" for t, x in U.groupby("tag")))
print("  GB tersetel per lipatan:  " + "  ".join(f"{t} {x.konfigurasi.iloc[0]}" for t, x in U.groupby("tag")))

# ================================================================ C3: rujukan dipilih di dalam lipatan
# isi disalin dari patokan_revisi_rujukan.py:31-43 (calon dan pilihan); baris luar-lipatan dari patokan_ambang_s.py,
# per kabupaten sudah dibatasi ke kabupaten positif di tahun latih lipatan (asalnya: ada_positif baku).
print("\n" + "=" * 110 + "\nC3. Rujukan dipilih di dalam lipatan (AUC-PR dalam-tahun luar-lipatan, tahun latih, label ulang)\n" + "=" * 110)
CALON = ["klimatologi", "klim_frek", "klim_frek1"]
O = L[L.sumber == "oof"]
beda_c3 = []
for t in B.BIASA:
    o = O[O.tag == t]
    nilai = {c: [] for c in CALON}
    for v, g in o.groupby("tahun_dalam"):
        y = g.y.to_numpy()
        if y.sum() == 0:
            continue
        for c in CALON:
            nilai[c].append(B.ap_cepat(y, g[c].to_numpy()))
    rata = {c: float(np.mean(v)) for c, v in nilai.items()}
    pilih = max(rata, key=rata.get)
    if pilih != REF:
        beda_c3.append(t)
    print(f"  S1 {t}: tahun dalam {o.tahun_dalam.min()}-{o.tahun_dalam.max()} ({len(nilai[CALON[0]])}); "
          + "  ".join(f"{B.LABEL[c]} {rata[c]:.3f}" for c in CALON) + f"  -> {B.LABEL[pilih]}"
          + ("" if pilih == REF else f"  (pasangan Tabel 2: {B.LABEL[REF]})"), flush=True)
print(f"  R10: rujukan per lipatan = pasangan Tabel 2 di keenam lipatan: {'ya' if not beda_c3 else 'TIDAK (' + ', '.join(beda_c3) + ')'}")

# ================================================================ S1: Tabel 2
print("\n" + "=" * 110 + f"\nS1. Tabel 2, ambang tahun latih (rujukan {B.LABEL[REF]})\n" + "=" * 110)
MODEL2 = ["klimatologi", "klim_frek", "klim_frek1", "log_klim_jeda", "log_ef1_jeda", "persistence", "seasonal_naive",
          "rasio", "fwi", "fwi_persentil", "fwi_logistik", "klim_fwi", "dc_nasional", "dc_persentil", "ffmc_nasional",
          "log_ef1_fdrs", "logistik", "xgb_sejarah", "rf", "xgb", "xgb_setel", "mlp"]
assert U[MODEL2].notna().all().all()
# ---- disalin dari patokan_revisi_tahun.py:90-101 tanpa perubahan isi (SEMUA -> MODEL2)
kab = U.gid.unique()
blok = {t: B.Blok(g, MODEL2, kab) for t, g in U.groupby("tag")}


def fn(b, j):
    y = b.y[j]
    return np.array([B.ap_cepat(y, b.s[m][j]) for m in MODEL2])


titik_th = {t: fn(b, slice(None)) for t, b in blok.items()}
titik = np.mean(list(titik_th.values()), axis=0)
rng = np.random.default_rng(B.SEED)
kb = B.ulang(blok, fn, 1000, False, rng, kab)
dt = B.ulang(blok, fn, 1000, True, rng, kab)
# ---- akhir salinan
ir = MODEL2.index(REF)
print(f"  {'model':<34}{'AP':>7}  {'unggul [kab]':<26}{'[dua tahap]':<20}{'[tahun t]':<20}{'n+':>3}{'pooled':>8}  "
      "per tahun 2020..2025")
for i, m in enumerate(MODEL2):
    pooled = B.ap_cepat(U.y.to_numpy(), U[m].to_numpy())
    if i == ir:
        print(f"  {NAMA[m] + ' (rujukan)':<34}{titik[i]:>7.3f}  {'—':<26}{'—':<20}{'—':<20}{'':>3}{pooled:>8.3f}", flush=True)
        continue
    pt = np.array([titik_th[t][i] - titik_th[t][ir] for t in B.BIASA])
    lk, hk = B.ci(kb[:, i] - kb[:, ir]); ld, hd = B.ci(dt[:, i] - dt[:, ir]); tl, tu = t95(pt)
    tanda = "  KEDUA SELANG TANPA NOL" if tanpa_nol(ld, hd) and tanpa_nol(tl, tu) else ""
    print(f"  {NAMA[m]:<34}{titik[i]:>7.3f}  {B.fmt(titik[i] - titik[ir], lk, hk):<26}[{ld:+.3f}, {hd:+.3f}]    "
          f"[{tl:+.3f}, {tu:+.3f}]    {int((pt > 0).sum()):>3}{pooled:>8.3f}  " + " ".join(f"{x:+.3f}" for x in pt) + tanda,
          flush=True)

# ================================================================ S2: Tabel 4
print("\n" + "=" * 110 + f"\nS2. Tabel 4, BSS dalam-tahun, ambang tahun latih (kalender = {B.LABEL[REF]})\n" + "=" * 110)
KOL = [REF, "rujukan_peluang", "ef1_tingkat", "konstan", "efd", "efld", "log_ef1_jeda", "rf", "xgb", "xgb_setel", "mlp",
       "xgb_nas", "xgb_nasdc"]
assert U[KOL].notna().all().all()
EMPAT = [REF, "rujukan_peluang", "ef1_tingkat", "efld"]
TIGA = [REF, "rujukan_peluang", "ef1_tingkat"]
SEL4 = [("konstan", [REF]), ("rujukan_peluang", [REF]), ("ef1_tingkat", [REF]), ("efd", TIGA), ("efld", TIGA),
        ("log_ef1_jeda", TIGA), ("rf", EMPAT), ("xgb", EMPAT), ("xgb_setel", TIGA), ("mlp", TIGA),
        ("xgb_nas", [REF, "ef1_tingkat", "efld"]), ("xgb_nasdc", EMPAT)]
# ---- disalin dari patokan_tingkat_o.py:183-203 dan 211-216 tanpa perubahan isi (g -> U)
g = U.reset_index(drop=True)
tags = list(B.BIASA)
kab = g.gid.unique()
pos = {k: i for i, k in enumerate(kab)}
ix = {c: i for i, c in enumerate(KOL)}
S = np.zeros((len(tags), len(kab), len(KOL)))
for a, t in enumerate(tags):
    x = g[g.tag == t]
    j = x.gid.map(pos).to_numpy(); y = x.y.to_numpy()
    for c in KOL:
        S[a, :, ix[c]] = np.bincount(j, weights=(x[c].to_numpy() - y) ** 2, minlength=len(kab))
th = S.sum(1)


def dua_tahap_cepat(n, rng):
    nt, nk = S.shape[0], S.shape[1]
    out = np.empty((n, nt, S.shape[2]))
    for r in range(n):
        pilih = rng.choice(nk, nk)
        tt = rng.integers(0, nt, nt)
        out[r] = S[:, pilih, :].sum(1)[tt]
    return out


def sel(R, a, r):
    pt = 1 - th[:, ix[a]] / th[:, ix[r]]
    rr = np.percentile((1 - R[:, :, ix[a]] / R[:, :, ix[r]]).mean(1), [2.5, 97.5])
    sr = 1 - th[:, ix[a]].sum() / th[:, ix[r]].sum()
    srq = np.percentile(1 - R[:, :, ix[a]].sum(1) / R[:, :, ix[r]].sum(1), [2.5, 97.5])
    return pt, rr, sr, srq
# ---- akhir salinan


RB = {b: dua_tahap_cepat(N, np.random.default_rng(b)) for b in BENIH}
NBARIS = np.array([(g.tag == t).sum() for t in tags])
print("  Brier rata-rata tahun: " + "  ".join(f"{'kalender' if c == REF else NAMA[c]} {(th[:, ix[c]] / NBARIS).mean():.5f}"
                                             for c in (REF, "rujukan_peluang", "ef1_tingkat", "konstan")))
print("  Brier per tahun (rata-rata baris): " + "  ".join(
    f"{t} " + "/".join(f"{th[a, ix[c]] / (g.tag == t).sum():.5f}" for c in (REF, "rujukan_peluang", "ef1_tingkat"))
    for a, t in enumerate(tags)) + "  (kalender/EF-P/EF-L)")
print(f"  {'prakiraan':<42}{'atas':<10}{'BSS':>8}  {'[dua tahap B.SEED]':<20}{'[tahun t]':<20} tanda  "
      "batas dua tahap benih 2026/1/2/3/4; per tahun")
HASIL4 = {}
for a, rs in SEL4:
    for r in rs:
        pt, rr, _, _ = sel(RB[B.SEED], a, r)
        tl, tu = t95(pt)
        vonis = []
        batas = []
        for b in BENIH:
            _, rb, _, _ = sel(RB[b], a, r)
            vonis.append(tanpa_nol(*rb)); batas.append(rb)
        lulus2, lulust = tanpa_nol(*rr), tanpa_nol(tl, tu)
        berubah = len(set(vonis)) > 1
        if lulus2 and lulust:
            tanda = "tebal" + (" (benih lain: dua tahap dengan nol)" if berubah else "")
        elif lulust:
            tanda = "tahun t saja"
        elif berubah:
            tanda = "‡"
        elif lulus2:
            tanda = "†"
        else:
            tanda = ""
        HASIL4[(a, r)] = (pt.mean(), rr, (tl, tu), tanda)
        nr = "kalender" if r == REF else NAMA[r]
        print(f"  {NAMA[a]:<42}{nr:<10}{pt.mean():>+8.3f}  [{rr[0]:+.3f}, {rr[1]:+.3f}]    [{tl:+.3f}, {tu:+.3f}]    "
              f"{tanda:<6} " + " ".join(f"[{x[0]:+.3f}, {x[1]:+.3f}]" for x in batas) + "  "
              + " ".join(f"{v:+.3f}" for v in pt), flush=True)

print("\n  Tabel 4 ringkas (sel naskah: BSS [dua tahap], tanda):")
print(f"  {'Forecast':<42}| Over calendar              | Over EF-P                  | Over EF-L                  | Over EF-LD")
for a, rs in SEL4:
    sel_teks = []
    for r in EMPAT:
        if (a, r) not in HASIL4:
            sel_teks.append(f"{'—':<26}")
            continue
        v, rr, _, tanda = HASIL4[(a, r)]
        s = f"{v:+.3f} [{rr[0]:+.3f}, {rr[1]:+.3f}]"
        s = f"**{s}**" if tanda.startswith("tebal") else s + ("" if tanda in ("", "tahun t saja") else tanda)
        sel_teks.append(f"{s:<26}")
    print(f"  {NAMA[a]:<42}| " + " | ".join(sel_teks))

# ================================================================ S3: Tabel 6 (tahun biasa, per kabupaten)
print("\n" + "=" * 110 + "\nS3. Tabel 6 kolom tahun biasa, nilai ekonomi relatif, ambang tahun latih\n" + "=" * 110)
if SAS != "y_kabupaten":
    print("  Tabel 6 hanya untuk sasaran per kabupaten.")
else:
    # ---- disalin dari patokan_revisi_tingkat.py:123-129 (cetak) dan 243-314 (fungsi siaga) tanpa perubahan isi
    def cetak(nama, pt, dist, kabd=None):
        lo, hi = np.nanpercentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
        kab = "" if kabd is None else " kab [{:+.4f}, {:+.4f}]".format(*np.nanpercentile(kabd, [2.5, 97.5]))
        lolos = lo > 0 and tl > 0
        print(f"    {nama:<40}{pt.mean():+.4f}{kab} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
              f"{'  LOLOS' if lolos else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
        return lolos

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

    QGRID = np.round(np.arange(0.02, 0.301, 0.02), 2)

    def hf(y, siaga):
        return (siaga & (y == 1)).sum() / max(y.sum(), 1), (siaga & (y == 0)).sum() / max((y == 0).sum(), 1)

    def kuota_setel(o, m, a):
        """bagian kuota bulanan dengan nilai rata-rata per tahun tertinggi di tahun latih."""
        tot = np.zeros(len(QGRID)); n = 0
        for _, g in o.groupby("tahun_dalam"):
            y = g.y.to_numpy()
            if y.sum() == 0:
                continue
            p = g[m].to_numpy(dtype=float); bl = g.bulan.to_numpy()
            urut = {mo: np.where(bl == mo)[0][np.argsort(-p[bl == mo], kind="stable")] for mo in np.unique(bl)}
            for qi, q in enumerate(QGRID):
                siaga = np.zeros(len(y), bool)
                for w in urut.values():
                    siaga[w[:max(1, int(round(q * len(w))))]] = True
                tot[qi] += nilai(*hf(y, siaga), y.mean(), a)
            n += 1
        return float(QGRID[int(np.argmax(tot / n))])

    def terbaik(y, p, a):
        calon = np.unique(p)
        pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
        H = 1 - np.searchsorted(pos, calon, side="left") / len(pos)
        F = 1 - np.searchsorted(neg, calon, side="left") / len(neg)
        return float(np.max(nilai(H, F, y.mean(), a)))
    # ---- akhir salinan

    # ---- disalin dari patokan_revisi_tingkat.py:319-372 dan 374-390 tanpa perubahan isi, kelompok "biasa" saja;
    # sumber baris latih: luar-lipatan (xgb, klim_frek1) dan EF-L baris latih dari patokan_ambang_s.py, label ulang.
    d = U
    kab = U.gid.unique()
    MK = ["xgb", "ef1_tingkat", "klim_frek1"]
    SETEL = ["xgb", "ef1_tingkat"]
    UK = ["hit", "FAR", "siaga"] + [f"REV{a}" for a in ALFA]
    nama = ([f"kuota {m} {u}" for m in MK for u in UK] + [f"amb10 {m} {u}" for m in MK for u in UK]
            + [f"{r} {m} REV{a}" for r in ("pa", "ambnilai") for m in MK for a in ALFA]
            + [f"setel {m} REV{a}" for m in SETEL for a in ALFA])
    iz = {n_: i for i, n_ in enumerate(nama)}
    kunci = [("S1", t) for t in B.BIASA]
    blk = {k: BlokB(d[d.tag == k[1]], MK, kab) for k in kunci}
    latih = {}
    for k in kunci:
        o = L[(L.sumber == "oof") & (L.tag == k[1])][["gid", "bulan", "tahun_dalam", "y", "xgb", "klim_frek1"]]
        e = L[(L.sumber == "efl") & (L.tag == k[1])][["gid", "bulan", "tahun_dalam", "y", "ef1_tingkat"]]
        assert not o.isna().any().any() and not e.isna().any().any()
        latih[k] = (o, e)
    sumber = lambda k, m: latih[k][1] if m == "ef1_tingkat" else latih[k][0]
    thr10 = {(k, m): np.quantile(sumber(k, m)[m].to_numpy(dtype=float), 0.9) for k in kunci for m in MK}
    thrv = {(k, m, a): ambang_oof(sumber(k, m), m, a) for k in kunci for m in MK for a in ALFA}
    qset = {(k, m, a): kuota_setel(sumber(k, m), m, a) for k in kunci for m in SETEL for a in ALFA}
    print(f"\n--- biasa: bagian kuota disetel (per lipatan, a = {ALFA}):")
    for m in SETEL:
        print(f"    {m:<12}" + "  ".join(f"{k[1]} " + "/".join(f"{qset[(k, m, a)]:.2f}" for a in ALFA) for k in kunci))
    for k, b in blk.items():
        b.kunci = k

    def fn(b, j):
        y = b.y[j]; out = []
        for m in MK:
            out += ukur(y, kuota(b, j, m))
        for m in MK:
            out += ukur(y, b.s[m][j] >= thr10[(b.kunci, m)])
        for m in MK:
            out += [nilai(*hf(y, b.s[m][j] > a), y.mean(), a) for a in ALFA]
        for m in MK:
            out += [nilai(*hf(y, b.s[m][j] >= thrv[(b.kunci, m, a)]), y.mean(), a) for a in ALFA]
        for m in SETEL:
            out += [nilai(*hf(y, kuota(b, j, m, qset[(b.kunci, m, a)])), y.mean(), a) for a in ALFA]
        return np.array(out, dtype=float)

    th2 = np.array([fn(blk[k], slice(None)) for k in kunci])
    t2 = th2.mean(0)
    print(f"    prevalensi {np.mean([blk[k].y.mean() for k in kunci]) * 100:.2f}%")
    print(f"    {'aturan':<26}{'hit':>7}{'FAR':>7}{'siaga':>8}   REV a = " + " / ".join(str(a) for a in ALFA))
    for aturan in ("kuota", "amb10"):
        for m in MK:
            print(f"    {aturan + ' ' + m:<26}{t2[iz[f'{aturan} {m} hit']]:>7.3f}{t2[iz[f'{aturan} {m} FAR']]:>7.3f}"
                  f"{t2[iz[f'{aturan} {m} siaga']] * 100:>7.1f}%   " + " / ".join(f"{t2[iz[f'{aturan} {m} REV{a}']]:+.3f}" for a in ALFA))
    for aturan, ms in (("pa", MK), ("ambnilai", MK), ("setel", SETEL)):
        for m in ms:
            print(f"    {aturan + ' ' + m:<26}{'':>22}   " + " / ".join(f"{t2[iz[f'{aturan} {m} REV{a}']]:+.3f}" for a in ALFA))
    TERBAIK = {}
    for m in MK:
        bt = np.array([[terbaik(blk[k].y, blk[k].s[m], a) for a in ALFA] for k in kunci]).mean(0)
        TERBAIK[m] = bt
        print(f"    {'terbaik ' + m:<26}{'':>22}   " + " / ".join(f"{v:+.3f}" for v in bt))
    print("    per tahun hit / siaga amb10 xgb: " + "  ".join(
        f"{k[1]} {x[iz['amb10 xgb hit']]:.3f}/{x[iz['amb10 xgb siaga']] * 100:.1f}%" for k, x in zip(kunci, th2)))
    R3 = B.dua_tahap(blk, fn, N, np.random.default_rng(B.SEED), kab)
    print("\n  R30. GB - EF-L, nilai ekonomi relatif")
    for aturan in ("pa", "ambnilai"):
        for a in ALFA:
            i, k = iz[f"{aturan} xgb REV{a}"], iz[f"{aturan} ef1_tingkat REV{a}"]
            cetak(f"{aturan} xgb - EF-L a={a}", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
    print("\n  R31. Kuota disetel - ambang nilai maksimum (aturan iv), model sama")
    for m in SETEL:
        for a in ALFA:
            i, k = iz[f"setel {m} REV{a}"], iz[f"ambnilai {m} REV{a}"]
            cetak(f"setel {m} - ambnilai a={a}", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
    print("\n  Tambahan: kuota 10 % hit GB - EF-L, kuota 10 % - kuota disetel GB")
    i, k = iz["kuota xgb hit"], iz["kuota ef1_tingkat hit"]
    cetak("hit kuota xgb - EF-L", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
    for a in ALFA:
        i, k = iz[f"kuota xgb REV{a}"], iz[f"setel xgb REV{a}"]
        cetak(f"kuota10 xgb - setel a={a}", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
    # ---- akhir salinan
    # kontras kuota - ambang (patokan_revisi_siaga2.py:107-112; R3 memakai tarikan yang sama dengan siaga2: benih B.SEED)
    print("\n  D7. Kuota 10 % - ambang (amb10 = kuantil 0,9 tahun latih; ambnilai = nilai maksimum tahun latih)")
    for m in ("xgb", "klim_frek1"):
        for a in ALFA:
            for lain in ("amb10", "ambnilai"):
                i, r = iz[f"kuota {m} REV{a}"], iz[f"{lain} {m} REV{a}"]
                cetak(f"kuota - {lain} {m} a={a}", th2[:, i] - th2[:, r], (R3[:, :, i] - R3[:, :, r]).mean(1))

    # Tambahan sesudah jalan pertama (Hasil_Patokan_25, penyimpangan 8): kontras yang dilaporkan kalimat §6.5 dan S14
    # naskah tetapi tidak dicetak di atas; tarikan R3 yang sama, tanpa prakiraan baru.
    print("\n  D8. GB - EF-1m (aturan peluang, kuota 10 % hit) dan selisih ambang terbaik")
    for a in ALFA:
        i, k = iz[f"pa xgb REV{a}"], iz[f"pa klim_frek1 REV{a}"]
        cetak(f"pa xgb - EF-1m a={a}", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
    i, k = iz["kuota xgb hit"], iz["kuota klim_frek1 hit"]
    cetak("hit kuota xgb - EF-1m", th2[:, i] - th2[:, k], (R3[:, :, i] - R3[:, :, k]).mean(1))
    print("    terbaik xgb - EF-1m: " + " / ".join(f"{x:+.3f}" for x in TERBAIK["xgb"] - TERBAIK["klim_frek1"]))

    print("\n  Tabel 6 ringkas, tahun biasa (a = " + " / ".join(str(a) for a in ALFA) + "):")
    BARIS6 = [("Prepare when p > a", "pa", MK), ("Threshold maximising value in training years", "ambnilai", MK),
              ("Fixed threshold, 10% of training-year district-months", "amb10", MK),
              ("Monthly quota, 10% of districts", "kuota", MK), ("Monthly quota, share tuned in training years", "setel", SETEL),
              ("Best threshold (bound)", "terbaik", MK)]
    for judul, aturan, ms in BARIS6:
        for n_, m in enumerate(ms):
            v = TERBAIK[m] if aturan == "terbaik" else [t2[iz[f"{aturan} {m} REV{a}"]] for a in ALFA]
            print(f"  {(judul if n_ == 0 else ''):<56}| {NAMA[m]:<18}| " + " / ".join(f"{x:+.3f}" for x in v))

# ================================================================ Cek: G2 / P
print("\n" + "=" * 110 + "\nCek G2 (Hasil_Patokan_12) dan P (Hasil_Patokan_22): BSS atas kalender, dua tahap benih B.SEED\n" + "=" * 110)
for a in ("rf", "xgb", "ef1_tingkat"):
    pt, rr, _, _ = sel(RB[B.SEED], a, REF)
    tl, tu = t95(pt)
    print(f"    BSS {a:<12} atas {'kalender':<12}{pt.mean():+.4f} 2tahap [{rr[0]:+.4f}, {rr[1]:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
          f"{'  LOLOS' if rr[0] > 0 and tl > 0 else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
print("selesai.")
