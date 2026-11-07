"""
PeatFireBench revisi fase C5 (Hasil_Patokan_8): kebocoran ambang sasaran terhadap rujukan sepadan.

Ambang dihitung ulang dari tahun latih tiap lipatan (sama dengan label_ulang di patokan_kepekaan_ambang.py:
persentil 90 kabupaten dengan batas bawah 10, dan persentil 90 gabungan, dari panel_nasional_bersih.csv).
rf, xgb, mlp dengan label ulang dari patokan_prediksi_kepekaan_penuh.csv (benih 42). EF-1m dan EF-3m
dihitung dari label ulang tahun latih. Dibandingkan dengan label baku (patokan_revisi_prediksi.csv, benih 42).

Pakai:    python -u patokan_revisi_ambang.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_ambang.log
"""
import numpy as np
import pandas as pd

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R

BATAS_BAWAH = 10
BELAJAR = ["rf", "xgb", "mlp"]
df = R.df
panel = pd.read_csv(f"{B.D}/panel_nasional_bersih.csv", usecols=["gid", "tahun", "titik_panas"])
semua = sorted(panel.tahun.unique())


def label_ulang(d, tahun_ambang):
    ref = panel[panel.tahun.isin(tahun_ambang)]
    q_kab = np.maximum(ref.groupby("gid").titik_panas.quantile(0.90), BATAS_BAWAH)
    gab = ref.titik_panas.quantile(0.90)
    d = d.copy()
    d["y_kabupaten"] = (d.titik_panas > d.gid.map(q_kab)).astype(int)
    d["y_gabungan"] = (d.titik_panas > gab).astype(int)
    return d


B.cek_ap()
baris = []
for sk, tahun_uji in (("S1", range(2020, 2026)), ("S2", (2015, 2014, 2019))):
    for th in tahun_uji:
        thn = [t for t in semua if (t < th if sk == "S1" else t != th)]
        d = label_ulang(df, thn)
        latih, uji = (d[d.tahun < th] if sk == "S1" else d[d.tahun != th]), d[d.tahun == th]
        for ylab in B.PASANGAN:
            b = uji[["gid", "bulan"]].copy()
            b["sasaran"], b["skema"], b["tag"], b["y_ulang"] = ylab, sk, str(th), uji[ylab].to_numpy()
            b["klim_frek1"] = R.klim_frek(latih, uji, ylab, 0); b["klim_frek"] = R.klim_frek(latih, uji, ylab, 1)
            pos = latih.groupby("gid")[ylab].sum()
            b["ada_lipatan"] = b.gid.map(pos).fillna(0).gt(0).astype(int).to_numpy()
            baris.append(b)
        print(f"{sk} {th}: ambang dari {thn[0]}-{thn[-1]}", flush=True)
U = pd.concat(baris, ignore_index=True)
K = pd.read_csv(f"{B.D}/patokan_prediksi_kepekaan_penuh.csv", dtype={"tag": str})
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
U = U.merge(K[kunci + ["y", "ada_positif"] + BELAJAR], on=kunci, how="inner", validate="one_to_one")
print(f"baris label ulang {len(U)}; y kepekaan = y dihitung ulang: {(U.y == U.y_ulang).mean():.4f}")
P = B.muat_P(fwi=False)


def tabel(d, ref, judul, n=1000):
    s1 = d[d.tag.isin(B.BIASA) & (d.skema == "S1")]
    kab = s1.gid.unique(); kol = BELAJAR + [ref]
    blok = {t: B.Blok(g, kol, kab) for t, g in s1.groupby("tag")}

    def fn(b, j):
        y = b.y[j]; r = B.ap_cepat(y, b.s[ref][j])
        return np.array([B.ap_cepat(y, b.s[m][j]) - r for m in BELAJAR] + [r])

    t0 = np.mean([fn(b, slice(None)) for b in blok.values()], axis=0)
    rng = np.random.default_rng(B.SEED)
    kb = B.ulang(blok, fn, n, False, rng, kab); dt = B.ulang(blok, fn, n, True, rng, kab)
    print(f"  {judul:<44} {len(kab)} kab, prev {s1.y.mean() * 100:.2f}%, AP {B.LABEL[ref]} {t0[-1]:.3f}")
    for i, m in enumerate(BELAJAR):
        print(f"      {m:<4} {B.fmt(t0[i], *B.ci(kb[:, i]))}  dua tahap [{B.ci(dt[:, i])[0]:+.3f}, {B.ci(dt[:, i])[1]:+.3f}]", flush=True)
    s2 = d[d.skema == "S2"]
    print("      S2: " + "; ".join(
        f"{t} xgb {np.mean([B.ap_cepat(g.y, g.xgb) - B.ap_cepat(g.y, g[ref]) for _, g in s2[s2.tag == t].groupby('tag')]):+.3f}"
        for t in B.EKSTREM))


print("=" * 100 + "\nC5. Tabel 2 (benih 42) dengan label baku dan label ulang per lipatan\n" + "=" * 100)
for ylab, ref in B.PASANGAN.items():
    print(f"\n--- {ylab}, rujukan {B.LABEL[ref]}")
    baku = B.ambil_sasaran(P, ylab)
    tabel(baku, ref, "label baku (ambang 2012-2025)")
    u = U[U.sasaran == ylab].copy(); u["y"] = u.y_ulang
    if ylab == "y_kabupaten":
        tabel(u[u.ada_positif == 1], ref, "label ulang, kabupaten tetap (ada_positif baku)")
        tabel(u[u.ada_lipatan == 1], ref, "label ulang, kabupaten per lipatan")
        print("      kabupaten per lipatan: " + "  ".join(
            f"{t} {u[(u.tag == t) & (u.skema == 'S1') & (u.ada_lipatan == 1)].gid.nunique()}" for t in B.BIASA))
    else:
        tabel(u, ref, "label ulang")
print("\nselesai.")
