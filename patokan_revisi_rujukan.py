"""
PeatFireBench revisi fase C3 (Hasil_Patokan_8): rujukan klimatologi dipilih di dalam lipatan, dari
prakiraan luar-lipatan pada tahun latih saja (calon MC, EF-3m, EF-1m), lalu Tabel 2 terhadap rujukan itu.

Pakai:    python -u patokan_revisi_rujukan.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_rujukan.log
"""
import numpy as np

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R

CALON = ["klimatologi", "klim_frek", "klim_frek1"]
MODEL = ["klimatologi", "klim_frek", "klim_frek1", "persistence", "seasonal_naive", "rasio", "fwi", "fwi_persentil",
         "fwi_logistik", "klim_fwi", "logistik", "log_klim_jeda", "rf", "xgb", "mlp"]
LIPATAN = [("S1", t) for t in range(2020, 2026)] + [("S2", t) for t in (2015, 2014, 2019)]
df = R.df
B.cek_ap()


def skor_calon(latih, uji, ylab):
    return {"klimatologi": R.klim_rata(latih, uji), "klim_frek": R.klim_frek(latih, uji, ylab, 1),
            "klim_frek1": R.klim_frek(latih, uji, ylab, 0)}


pilihan = {}
print("=" * 100 + "\nC3. Pilihan rujukan di dalam lipatan (AUC-PR dalam-tahun luar-lipatan pada tahun latih)\n" + "=" * 100)
for ylab in B.PASANGAN:
    print(f"\n--- {ylab}")
    for sk, th in LIPATAN:
        thn = sorted(t for t in df.tahun.unique() if (t < th if sk == "S1" else t != th))
        latih_luar = df[df.tahun.isin(thn)]
        nilai = {c: [] for c in CALON}
        for v in thn:
            latih, uji = latih_luar[latih_luar.tahun != v], latih_luar[latih_luar.tahun == v]
            sk_ = skor_calon(latih, uji, ylab)
            mask = (uji.ada_positif == 1).to_numpy() if ylab == "y_kabupaten" else np.ones(len(uji), bool)
            y = uji[ylab].to_numpy()[mask]
            if y.sum() == 0:
                continue
            for c in CALON:
                nilai[c].append(B.ap_cepat(y, sk_[c][mask]))
        rata = {c: float(np.mean(v)) for c, v in nilai.items()}
        pilih = max(rata, key=rata.get)
        pilihan[(ylab, sk, str(th))] = pilih
        print(f"  {sk} {th}: tahun dalam {thn[0]}-{thn[-1]} ({len(nilai[CALON[0]])}); "
              + "  ".join(f"{B.LABEL[c]} {rata[c]:.3f}" for c in CALON) + f"  -> {B.LABEL[pilih]}"
              + ("" if pilih == B.PASANGAN[ylab] else f"  (pasangan Tabel 2: {B.LABEL[B.PASANGAN[ylab]]})"), flush=True)

P = B.muat_P()
P["rujukan_lipatan"] = np.nan
for (ylab, sk, t), c in pilihan.items():
    m = (P.sasaran == ylab) & (P.skema == sk) & (P.tag == t)
    P.loc[m, "rujukan_lipatan"] = P.loc[m, c]
assert P.loc[P.tag.isin(B.BIASA) | (P.skema == "S2"), "rujukan_lipatan"].notna().all()

print("\n" + "=" * 100 + "\nTabel 2 terhadap rujukan per lipatan (S1 2020-2025)\n" + "=" * 100)
for ylab in B.PASANGAN:
    d = B.ambil_sasaran(P, ylab)
    s1 = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
    kol = MODEL + ["rujukan_lipatan"]
    kab = s1.gid.unique()
    blok = {t: B.Blok(g, kol, kab) for t, g in s1.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.array([B.ap_cepat(y, b.s[m][j]) for m in kol])

    titik = np.mean([fn(b, slice(None)) for b in blok.values()], axis=0)
    rng = np.random.default_rng(B.SEED)
    kb = B.ulang(blok, fn, 1000, False, rng, kab); dt = B.ulang(blok, fn, 1000, True, rng, kab)
    ir = kol.index("rujukan_lipatan")
    print(f"\n--- {ylab}: rujukan per lipatan AP {titik[ir]:.3f} (pasangan Tabel 2 {B.LABEL[B.PASANGAN[ylab]]} "
          f"{titik[kol.index(B.PASANGAN[ylab])]:.3f})")
    for i, m in enumerate(MODEL):
        print(f"  {m:<15}{titik[i]:>7.3f}  {B.fmt(titik[i] - titik[ir], *B.ci(kb[:, i] - kb[:, ir]))}  dua tahap "
              f"[{B.ci(dt[:, i] - dt[:, ir])[0]:+.3f}, {B.ci(dt[:, i] - dt[:, ir])[1]:+.3f}]", flush=True)
    s2 = d[d.skema == "S2"]
    print("  S2 per tahun, atas rujukan per lipatan [kab 500]:")
    for t in B.EKSTREM:
        g = s2[s2.tag == t]; kk = g.gid.unique(); bl = {t: B.Blok(g, ["xgb", "rf", "mlp", "rujukan_lipatan"], kk)}

        def f2(b, j):
            y = b.y[j]; r = B.ap_cepat(y, b.s["rujukan_lipatan"][j])
            return np.array([B.ap_cepat(y, b.s[m][j]) - r for m in ("xgb", "rf", "mlp")])
        p0 = f2(bl[t], slice(None)); bs = B.ulang(bl, f2, 500, False, np.random.default_rng(B.SEED), kk)
        print(f"    {t} ({B.LABEL[pilihan[(ylab, 'S2', t)]]}): " + "; ".join(
            f"{m} {B.fmt(p0[i], *B.ci(bs[:, i]))}" for i, m in enumerate(("xgb", "rf", "mlp"))))
print("\nselesai.")
