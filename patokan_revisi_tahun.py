"""
PeatFireBench revisi fase C1 dan C2 (Hasil_Patokan_8): ketidakpastian antartahun untuk Tabel 2 dan
kontras tahun ekstrem pada pasangan rujukan Tabel 2 (EF-1m per kabupaten, EF-3m nasional).

Pakai:    python -u patokan_revisi_tahun.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_tahun.log
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

import patokan_revisi_bantu as B

MODEL = ["klimatologi", "klim_frek", "klim_frek1", "persistence", "seasonal_naive", "rasio", "fwi", "fwi_persentil",
         "fwi_logistik", "klim_fwi", "logistik", "log_klim_jeda", "rf", "xgb", "mlp"]
BELAJAR = ["rf", "xgb", "mlp"]
B.cek_ap()
P = B.muat_P()

# rata-rata enam benih (S1 saja) sebagai baris kepekaan
Bn = pd.read_csv(f"{B.D}/patokan_benih.csv", dtype={"tag": str})
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = P.merge(Bn[kunci + [f"{m}_{i}" for m in BELAJAR for i in range(1, 6)]], on=kunci, how="left", validate="one_to_one")
for m in BELAJAR:
    P[m + "6"] = P[[m] + [f"{m}_{i}" for i in range(1, 6)]].mean(axis=1)
SEMUA = MODEL + [m + "6" for m in BELAJAR]

print("=" * 100 + "\nC1. Tabel 2 dengan ketidakpastian antartahun (S1 2020-2025)\n" + "=" * 100)
for sas, ref in B.PASANGAN.items():
    d = B.ambil_sasaran(P, sas)
    d = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
    assert d[SEMUA].notna().all().all()
    kab = d.gid.unique()
    blok = {t: B.Blok(g, SEMUA, kab) for t, g in d.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.array([B.ap_cepat(y, b.s[m][j]) for m in SEMUA])

    titik_th = {t: fn(b, slice(None)) for t, b in blok.items()}
    titik = np.mean(list(titik_th.values()), axis=0)
    rng = np.random.default_rng(B.SEED)
    kb = B.ulang(blok, fn, 1000, False, rng, kab)
    dt = B.ulang(blok, fn, 1000, True, rng, kab)
    ir = SEMUA.index(ref); i3 = SEMUA.index("klim_frek"); imc = SEMUA.index("klimatologi")
    print(f"\n--- {sas}: rujukan {B.LABEL[ref]} ({ref}), {len(kab)} kabupaten, AP rujukan {titik[ir]:.3f}")
    print(f"  {'model':<15}{'AP':>7}  {'unggul [kab]':<26}{'[dua tahap]':<18}{'t(df5)':>7} {'n+':>3}  per tahun 2020..2025"
          + ("   unggul atas EF-3m [kab] [dua tahap]" if sas == "y_kabupaten" else ""))
    for i, m in enumerate(SEMUA):
        if i == ir:
            continue
        g = titik[i] - titik[ir]
        pt = np.array([titik_th[t][i] - titik_th[t][ir] for t in B.BIASA])
        tt = pt.mean() / (pt.std(ddof=1) / np.sqrt(len(pt)))
        lk, hk = B.ci(kb[:, i] - kb[:, ir]); ld, hd = B.ci(dt[:, i] - dt[:, ir])
        s = (f"  {m:<15}{titik[i]:>7.3f}  {B.fmt(g, lk, hk):<26}[{ld:+.3f}, {hd:+.3f}]  {tt:>6.2f} {int((pt > 0).sum()):>3}  "
             + " ".join(f"{x:+.3f}" for x in pt))
        if sas == "y_kabupaten" and i != i3:
            g3 = titik[i] - titik[i3]
            s += f"   {B.fmt(g3, *B.ci(kb[:, i] - kb[:, i3]))} [{B.ci(dt[:, i] - dt[:, i3])[0]:+.3f}, {B.ci(dt[:, i] - dt[:, i3])[1]:+.3f}]"
        print(s, flush=True)
    sel = titik[ir] - titik[imc]
    print(f"  selisih keunggulan (atas MC - atas {B.LABEL[ref]}) = AP({B.LABEL[ref]}) - AP(MC), sama untuk semua model: "
          f"{B.fmt(sel, *B.ci(kb[:, ir] - kb[:, imc]))}, dua tahap [{B.ci(dt[:, ir] - dt[:, imc])[0]:+.3f}, {B.ci(dt[:, ir] - dt[:, imc])[1]:+.3f}]")
    print(f"    per tahun: " + " ".join(f"{titik_th[t][ir] - titik_th[t][imc]:+.3f}" for t in B.BIASA))
    for m in BELAJAR:
        i = SEMUA.index(m)
        r0 = (titik[i] - titik[imc]) / (titik[i] - titik[ir])
        rk = (kb[:, i] - kb[:, imc]) / (kb[:, i] - kb[:, ir]); rd = (dt[:, i] - dt[:, imc]) / (dt[:, i] - dt[:, ir])
        print(f"    rasio lama {m}: {r0:.2f}; persentil kab [{B.ci(rk)[0]:.2f}, {B.ci(rk)[1]:.2f}], dua tahap "
              f"[{B.ci(rd)[0]:.2f}, {B.ci(rd)[1]:.2f}]; bagian ulangan dua tahap dengan penyebut <= 0: {(dt[:, i] - dt[:, ir] <= 0).mean():.3f}")


print("\n" + "=" * 100 + "\nC2. Kontras ekstrem - tenang pada pasangan Tabel 2, 411 kabupaten (seperti A2)\n" + "=" * 100)
Pk = P[P.ada_positif == 1]
kab = Pk.gid.unique()


def blok_kontras(tenang):
    out = {}
    for sas, ref in B.PASANGAN.items():
        for sk, tags in (("S2", B.EKSTREM), ("S1", tenang)):
            for t in tags:
                g = Pk[(Pk.sasaran == sas) & (Pk.skema == sk) & (Pk.tag == t)]
                out[(sas, sk, t)] = B.Blok(g, BELAJAR + [ref], kab)
    return out


def stat(b, j, ref):
    y = b.y[j]
    if y.sum() == 0 or y.sum() == len(y):
        return np.full(2 * len(BELAJAR), np.nan)
    r_ap, r_auc = B.ap_cepat(y, b.s[ref][j]), auc(y, b.s[ref][j])
    return np.array([B.ap_cepat(y, b.s[m][j]) - r_ap for m in BELAJAR] + [auc(y, b.s[m][j]) - r_auc for m in BELAJAR])


def kontras(G, sas, eks, ten):
    return np.nanmean([G[(sas, "S2", t)] for t in eks], axis=0) - np.nanmean([G[(sas, "S1", t)] for t in ten], axis=0)


def jalan(tenang, nama, n=1000, n_kab=300):
    bl = blok_kontras(tenang)
    G0 = {k: stat(b, slice(None), B.PASANGAN[k[0]]) for k, b in bl.items()}
    rng = np.random.default_rng(B.SEED)
    kb, dt = [], []
    for i in range(n):
        pilih = rng.choice(kab, len(kab))
        G = {k: stat(b, b.baris(pilih), B.PASANGAN[k[0]]) for k, b in bl.items()}
        eks = [B.EKSTREM[j] for j in rng.integers(0, 3, 3)]
        ten = [tenang[j] for j in rng.integers(0, len(tenang), len(tenang))]
        k1 = [kontras(G, s, B.EKSTREM, tenang) for s in B.PASANGAN]
        k2 = [kontras(G, s, eks, ten) for s in B.PASANGAN]
        if i < n_kab:
            kb.append([k1[0], k1[1], k1[0] - k1[1]])
        dt.append([k2[0], k2[1], k2[0] - k2[1]])
    kb, dt = np.array(kb), np.array(dt)
    t0 = [kontras(G0, s, B.EKSTREM, tenang) for s in B.PASANGAN]
    print(f"\n--- tenang = {nama}; selang kabupaten {n_kab}, dua tahap {n}")
    for u, uk in enumerate(("m1 AUC-PR", "m4 ROC-AUC")):
        for mi, m in enumerate(BELAJAR):
            c = u * len(BELAJAR) + mi
            print(f"  {uk:<10} {m:<4} per kab {B.fmt(t0[0][c], *B.ci(kb[:, 0, c]))}  nasional {B.fmt(t0[1][c], *B.ci(kb[:, 1, c]))}  "
                  f"selisih {t0[0][c] - t0[1][c]:+.3f} kab [{B.ci(kb[:, 2, c])[0]:+.3f}, {B.ci(kb[:, 2, c])[1]:+.3f}] "
                  f"dua tahap [{B.ci(dt[:, 2, c])[0]:+.3f}, {B.ci(dt[:, 2, c])[1]:+.3f}]", flush=True)
    print("  per tahun xgb m1 (atas rujukan pasangan):")
    for s in B.PASANGAN:
        print(f"    {s:<12} " + "  ".join(f"{k[1]}{'e' if k[1] in B.EKSTREM and k[0] == 'S2' else ''} {G0[k][1]:+.3f}"
                                         for k in bl if k[0] == s))


jalan(B.TENANG, "S1 2020, 2021, 2022, 2024, 2025 (seperti A2)")
jalan(B.TENANG + ["2023"], "kepekaan: termasuk 2023", n=500, n_kab=200)
print("\nselesai.")
