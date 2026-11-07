"""
PeatFireBench revisi fase A — analisis (Hasil_Patokan_6, A1-A5; aturan baca R1-R5 di berkas itu).

Masukan: DL_FIRE_NASIONAL/patokan_revisi_prediksi.csv, DL_FIRE_NASIONAL/patokan_revisi_oof.csv
Pakai:   python patokan_revisi_analisis.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_analisis.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str})
O = pd.read_csv(os.path.join(D, "patokan_revisi_oof.csv"), dtype={"tag": str})
O = O[O.ada_positif == 1]
SAS = ("y_kabupaten", "y_gabungan")
BIASA = [str(t) for t in range(2020, 2026)]
EKSTREM = [("S2", "2015"), ("S2", "2014"), ("S2", "2019")]
TENANG = [("S1", t) for t in ("2020", "2021", "2022", "2024", "2025")]
TENANG_Q = [("Q", t) for t in ("2020", "2021", "2022", "2024", "2025")]
REF = "klim_frek"
MODEL_TABEL = ["klim_frek", "klim_frek1", "klimatologi", "log_klim_jeda", "persistence", "seasonal_naive", "rasio",
               "fwi", "fwi_logistik", "logistik", "rf", "xgb", "mlp"]
BELAJAR = ["xgb", "rf", "mlp"]
SEED = 2026


def fmt(x, lo, hi):
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def indeks(blok, kab):
    g = blok.gid.to_numpy()
    return {k: np.where(g == k)[0] for k in kab}


# ---------------------------------------------------------------- A1, A3
def dalam_tahun(d, a, b=None):
    v = [ap(g.y, g[a]) - (ap(g.y, g[b]) if b else 0.0) for _, g in d.groupby("tag") if g.y.sum()]
    return float(np.mean(v))


def boot_dalam_tahun(d, a, b, n=1000):
    uk = d.gid.unique(); rng = np.random.default_rng(SEED)
    per = {t: g for t, g in d.groupby("tag")}
    idx = {t: indeks(g, uk) for t, g in per.items()}
    arr = {t: (g.y.to_numpy(), g[a].to_numpy(dtype=float), g[b].to_numpy(dtype=float)) for t, g in per.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, len(uk)); v = []
        for t in per:
            j = np.concatenate([idx[t][k] for k in pilih]); y, sa, sb = (x[j] for x in arr[t])
            if y.sum():
                v.append(ap(y, sa) - ap(y, sb))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


print("=" * 100 + "\nA1/A3. Tabel 2 dan 3 ulang terhadap klim_frek (rujukan baru)\n" + "=" * 100)
for sas in SAS:
    d0 = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    biasa = d0[(d0.skema == "S1") & d0.tag.isin(BIASA)]
    print(f"\n--- {sas}: {biasa.gid.nunique()} satuan, S1 2020-2025 {len(biasa)} baris, prevalensi {biasa.y.mean() * 100:.2f}%")
    print(f"  {'model':<16}{'AUC-PR dlm-thn':>15}{'unggul vs klim_frek [95%]':>32}{'unggul vs klim rata-rata':>26}")
    for m in sorted(MODEL_TABEL, key=lambda m: -dalam_tahun(biasa, m)):
        u = fmt(dalam_tahun(biasa, m, REF), *boot_dalam_tahun(biasa, m, REF)) if m != REF else ""
        print(f"  {m:<16}{dalam_tahun(biasa, m):>15.3f}{u:>32}{dalam_tahun(biasa, m, 'klimatologi'):>+26.3f}")
    print(f"  A3: xgb - log_klim_jeda {fmt(dalam_tahun(biasa, 'xgb', 'log_klim_jeda'), *boot_dalam_tahun(biasa, 'xgb', 'log_klim_jeda'))}")
    print(f"      rf  - log_klim_jeda {fmt(dalam_tahun(biasa, 'rf', 'log_klim_jeda'), *boot_dalam_tahun(biasa, 'rf', 'log_klim_jeda'))}")
    s2 = d0[d0.skema == "S2"]
    print("  S2 tahun ekstrem disisihkan, unggul atas klim_frek [95%]:")
    for _, t in EKSTREM:
        dd = s2[s2.tag == t]
        teks = [f"{m} {fmt(dalam_tahun(dd, m, REF), *boot_dalam_tahun(dd, m, REF, n=500))}"
                for m in ("xgb", "rf", "mlp", "log_klim_jeda", "fwi_logistik")]
        print(f"    {t} (prev {dd.y.mean() * 100:.1f}%, klim_frek {ap(dd.y, dd[REF]):.3f}, klim rata {ap(dd.y, dd.klimatologi):.3f}): " + "; ".join(teks))


# ---------------------------------------------------------------- A2
def statistik(y, s, r, brier):
    """kelima ukuran keunggulan s atas rujukan r pada satu blok; m5 hanya bila r peluang."""
    if y.sum() == 0 or y.sum() == len(y):
        return [np.nan] * 5
    pi = y.mean(); a_m, a_r = ap(y, s), ap(y, r)
    m5 = 1 - np.mean((s - y) ** 2) / np.mean((r - y) ** 2) if brier else np.nan
    return [a_m - a_r, (a_m - a_r) / (1 - a_r), (a_m - a_r) / (1 - pi), auc(y, s) - auc(y, r), m5]


UKURAN = ["m1 AUC-PR mentah", "m2 ruang sisa", "m3 AUC-PR ternormalisasi", "m4 ROC-AUC", "m5 Brier skill"]


def uji_temuan1(ref, tenang, nama_lengan, n=1000, n_kab=300):
    Pk = P[P.ada_positif == 1]
    kunci = EKSTREM + tenang
    blok = {(s,) + k: Pk[(Pk.sasaran == s) & (Pk.skema == k[0]) & (Pk.tag == k[1])] for s in SAS for k in kunci}
    kab = Pk.gid.unique()
    idx = {k: indeks(b, kab) for k, b in blok.items()}
    arr = {k: (b.y.to_numpy(), {m: b[m].to_numpy(dtype=float) for m in BELAJAR + [ref]}) for k, b in blok.items()}

    def gain_semua(pilih):
        out = {}
        for k, (y, sk) in arr.items():
            j = np.concatenate([idx[k][g] for g in pilih]) if pilih is not None else slice(None)
            for m in BELAJAR:
                out[k + (m,)] = statistik(y[j], sk[m][j], sk[ref][j], ref == REF)
        return out

    def kontras(G, s, m, eks, ten):
        e = np.nanmean([G[(s,) + k + (m,)] for k in eks], axis=0)
        q = np.nanmean([G[(s,) + k + (m,)] for k in ten], axis=0)
        return e - q

    G0 = gain_semua(None)
    rng = np.random.default_rng(SEED)
    kab_bs, dua_bs = [], []
    for i in range(n):
        G = gain_semua(rng.choice(kab, len(kab)))
        eks = [EKSTREM[j] for j in rng.integers(0, len(EKSTREM), len(EKSTREM))]
        ten = [tenang[j] for j in rng.integers(0, len(tenang), len(tenang))]
        for m in BELAJAR:
            k1 = [kontras(G, s, m, EKSTREM, tenang) for s in SAS]
            k2 = [kontras(G, s, m, eks, ten) for s in SAS]
            if i < n_kab:
                kab_bs.append((m, k1[0], k1[1], k1[0] - k1[1]))
            dua_bs.append((m, k2[0], k2[1], k2[0] - k2[1]))
    print(f"\n--- rujukan {ref}; tenang = {nama_lengan}; selang kabupaten {n_kab} ulangan, dua tahap {n} ulangan")
    for m in BELAJAR:
        titik = [kontras(G0, s, m, EKSTREM, tenang) for s in SAS]
        kb = np.array([r[1:] for r in kab_bs if r[0] == m]); db = np.array([r[1:] for r in dua_bs if r[0] == m])
        print(f"  {m}")
        for u, nama in enumerate(UKURAN):
            if np.isnan(titik[0][u]):
                continue
            sel = titik[0][u] - titik[1][u]
            lo_k, hi_k = np.nanpercentile(kb[:, 2, u], [2.5, 97.5]); lo_d, hi_d = np.nanpercentile(db[:, 2, u], [2.5, 97.5])
            pk = [np.nanpercentile(kb[:, i, u], [2.5, 97.5]) for i in (0, 1)]
            print(f"    {nama:<26} per kab {fmt(titik[0][u], *pk[0])}  nasional {fmt(titik[1][u], *pk[1])}  "
                  f"selisih {sel:+.3f} kab [{lo_k:+.3f}, {hi_k:+.3f}] dua-tahap [{lo_d:+.3f}, {hi_d:+.3f}]")
    print("  per tahun, xgb (m1 / m2 / m5):")
    for s in SAS:
        for k in EKSTREM + tenang:
            g = G0[(s,) + k + ("xgb",)]
            print(f"    {s:<12} {k[0]} {k[1]}  m1 {g[0]:+.3f}  m2 {g[1]:+.3f}  m5 {g[4]:+.3f}  prev {blok[(s,) + k].y.mean() * 100:5.1f}%")


print("\n" + "=" * 100 + "\nA2. Temuan 1 ulang: kontras ekstrem - tenang, 411 kabupaten\n" + "=" * 100)
uji_temuan1(REF, TENANG, "S1 kronologis (seperti naskah)")
uji_temuan1(REF, TENANG_Q, "Q disisihkan (rezim latih sama dengan S2)")
uji_temuan1("klimatologi", TENANG, "S1 kronologis; rujukan versi terkirim", n=300)


# ---------------------------------------------------------------- A4, A5
ALFA = [0.05, 0.1, 0.2, 0.3, 0.5]
MOD_KEP = ["xgb", "klim_frek", "klimatologi", "log_klim_jeda"]
KEL = {"biasa (S1 2020-2025)": [("S1", t) for t in BIASA], "ekstrem (S2 2014, 2015, 2019)": EKSTREM}


def nilai(H, F, s, a):
    e_klim, e_semp = min(a, s), a * s
    return (e_klim - (F * a * (1 - s) + H * a * s + (1 - H) * s)) / (e_klim - e_semp)


def nilai_ambang(y, p, thr, a):
    siaga = p >= thr
    H = (siaga & (y == 1)).sum() / max(y.sum(), 1); F = (siaga & (y == 0)).sum() / max((y == 0).sum(), 1)
    return nilai(H, F, y.mean(), a)


def kurva_roc(y, p):
    o = np.argsort(-p, kind="stable"); ys, ps = y[o], p[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    ujung = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]
    return np.r_[0, tp[ujung] / max(ys.sum(), 1)], np.r_[0, fp[ujung] / max((1 - ys).sum(), 1)]


def amplop(y, p, a):
    H, F = kurva_roc(y, p)
    return nilai(H, F, y.mean(), a).max()


def ambang_oof(o, m, a):
    """ambang yang memaksimalkan rata-rata nilai per tahun dalam pada prakiraan luar-lipatan."""
    calon = np.unique(np.quantile(o[m].to_numpy(dtype=float), np.linspace(0, 1, 401)))
    per = [(g.y.to_numpy(), g[m].to_numpy(dtype=float)) for _, g in o.groupby("tahun_dalam") if g.y.sum()]
    skor = [np.mean([nilai_ambang(y, p, c, a) for y, p in per]) for c in calon]
    return calon[int(np.argmax(skor))]


Pt = P[(P.sasaran == "y_kabupaten") & (P.ada_positif == 1)]
uji = {k: Pt[(Pt.skema == k[0]) & (Pt.tag == k[1])] for v in KEL.values() for k in v}
oof = {k: O[(O.skema == k[0]) & (O.tag == k[1])] for v in KEL.values() for k in v}
thr = {(k, m, a): ambang_oof(oof[k], m, a) for k in uji for m in MOD_KEP for a in ALFA}

print("\n" + "=" * 100 + "\nA4. Nilai keputusan dengan ambang dari tahun latih (luar-lipatan), per kabupaten\n" + "=" * 100)
rng = np.random.default_rng(SEED)
for kel, kunci in KEL.items():
    print(f"\n--- {kel}")
    for m in MOD_KEP:
        v = [np.mean([nilai_ambang(uji[k].y.to_numpy(), uji[k][m].to_numpy(dtype=float), thr[(k, m, a)], a) for k in kunci]) for a in ALFA]
        e = [np.mean([amplop(uji[k].y.to_numpy(), uji[k][m].to_numpy(dtype=float), a) for k in kunci]) for a in ALFA]
        print(f"  {m:<14} dapat dicapai " + " ".join(f"{x:+.3f}" for x in v) + "   amplop " + " ".join(f"{x:+.3f}" for x in e))
    kab = Pt.gid.unique(); idx = {k: indeks(uji[k], kab) for k in kunci}
    for a in ALFA:
        def stat(pilih):
            out = []
            for k in kunci:
                j = np.concatenate([idx[k][g] for g in pilih]) if pilih is not None else slice(None)
                y = uji[k].y.to_numpy()[j]
                out.append(nilai_ambang(y, uji[k].xgb.to_numpy()[j], thr[(k, "xgb", a)], a)
                           - nilai_ambang(y, uji[k][REF].to_numpy()[j], thr[(k, REF, a)], a))
            return np.mean(out)
        bs = [stat(rng.choice(kab, len(kab))) for _ in range(500)]
        print(f"  a = {a}: xgb - klim_frek dapat dicapai {fmt(stat(None), *np.percentile(bs, [2.5, 97.5]))}")

print("\n" + "=" * 100 + "\nA5. Aturan siaga kausal: hit rate / FAR / bagian siaga terwujud (rata-rata tahun)\n" + "=" * 100)


def ukur(y, siaga):
    tp = (siaga & (y == 1)).sum()
    return tp / max(y.sum(), 1), 1 - tp / max(siaga.sum(), 1), siaga.mean()


for kel, kunci in KEL.items():
    print(f"\n--- {kel}")
    print("  (a) ambang = skor yang menyiagakan 10 % prakiraan luar-lipatan tahun latih")
    for m in MOD_KEP:
        r = np.array([ukur(uji[k].y.to_numpy(), uji[k][m].to_numpy(dtype=float) >= np.quantile(oof[k][m].to_numpy(dtype=float), 0.9))
                      for k in kunci])
        print(f"    {m:<14} hit {r[:, 0].mean():.3f}  FAR {r[:, 1].mean():.3f}  siaga {r[:, 2].mean() * 100:.1f}%  (siaga per tahun "
              + " ".join(f"{x * 100:.0f}%" for x in r[:, 2]) + ")")
    print("  (b) 10 % kabupaten tertinggi tiap bulan")
    for m in MOD_KEP + ["rf", "mlp"]:
        r = []
        for k in kunci:
            d = uji[k]; siaga = np.zeros(len(d), bool)
            for _, g in d.groupby("bulan"):
                n = max(1, int(round(0.1 * len(g))))
                siaga[d.index.get_indexer(g.index[np.argsort(-g[m].to_numpy(dtype=float), kind="stable")[:n]])] = True
            r.append(ukur(d.y.to_numpy(), siaga))
        r = np.array(r)
        print(f"    {m:<14} hit {r[:, 0].mean():.3f}  FAR {r[:, 1].mean():.3f}  siaga {r[:, 2].mean() * 100:.1f}%")
print("\nselesai.")
