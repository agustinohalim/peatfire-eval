"""
PeatFireBench revisi fase D9 (Hasil_Patokan_9): gradient boosting dengan penyetelan di dalam lipatan.

Grid 18: max_depth {2, 3, 5} x n_estimators {150, 300, 600} x min_child_weight {1, 10}; learning_rate
0,05, subsample dan colsample 0,9, reg_lambda 1, benih 42, fitur penuh (sama dengan patokan_revisi_prediksi).
n_estimators dinilai dari satu model 600 pohon dengan iteration_range (setara melatih 150/300/600 pohon
dengan benih sama, karena pohon ditambah berurutan). Validasi dalam: dua tahun latih terbaru; pilihan
menurut AUC-PR dalam-tahun rata-rata di validasi (per kabupaten: kabupaten ada_positif saja). Dilatih
ulang pada semua tahun latih dengan konfigurasi terpilih. Tahun uji tidak dipakai untuk memilih.

Pakai:    python -u patokan_revisi_setel.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_setel.log
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_setel.csv (tidak masuk repo)
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R

warnings.filterwarnings("ignore")
OUT = f"{B.D}/patokan_revisi_setel.csv"
DEPTH, NEST, MCW = (2, 3, 5), (150, 300, 600), (1, 10)
df = R.df


def model(depth, mcw, n=600):
    return XGBClassifier(n_estimators=n, max_depth=depth, min_child_weight=mcw, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)


def ap_val(d, p, ylab):
    v = []
    for _, g in d.assign(p=p).groupby("tahun"):
        if ylab == "y_kabupaten":
            g = g[g.ada_positif == 1]
        if g[ylab].sum() > 0:
            v.append(B.ap_cepat(g[ylab].to_numpy(), g.p.to_numpy()))
    return np.mean(v)


if not os.path.exists(OUT):
    baris = []
    for sk, tahun in (("S1", range(2020, 2026)), ("S2", (2015, 2014, 2019))):
        for th in tahun:
            latih = df[df.tahun < th] if sk == "S1" else df[df.tahun != th]
            uji = df[df.tahun == th]
            val_th = sorted(latih.tahun.unique())[-2:]
            dalam, val = latih[~latih.tahun.isin(val_th)], latih[latih.tahun.isin(val_th)]
            for ylab in B.PASANGAN:
                skor = {}
                for dp in DEPTH:
                    for mc in MCW:
                        m = model(dp, mc).fit(dalam[R.FITUR].fillna(0.0).to_numpy(), dalam[ylab].to_numpy())
                        for n in NEST:
                            p = m.predict_proba(val[R.FITUR].fillna(0.0).to_numpy(), iteration_range=(0, n))[:, 1]
                            skor[(dp, n, mc)] = ap_val(val, p, ylab)
                best = max(skor, key=skor.get)
                m = model(best[0], best[2], best[1]).fit(latih[R.FITUR].fillna(0.0).to_numpy(), latih[ylab].to_numpy())
                b = uji[["gid", "bulan"]].copy()
                b["sasaran"], b["skema"], b["tag"] = ylab, sk, str(th)
                b["xgb_setel"] = m.predict_proba(uji[R.FITUR].fillna(0.0).to_numpy())[:, 1]
                b["konfigurasi"] = f"d{best[0]}_n{best[1]}_mcw{best[2]}"
                baris.append(b)
                print(f"{sk} {th} {ylab}: validasi {val_th}, terpilih depth {best[0]} n {best[1]} mcw {best[2]} "
                      f"(AP val {skor[best]:.3f}; baku d3 n300 mcw1 {skor[(3, 300, 1)]:.3f})", flush=True)
    pd.concat(baris, ignore_index=True).to_csv(OUT, index=False)

N = pd.read_csv(OUT, dtype={"tag": str})
B.cek_ap()
P = B.muat_P(fwi=False)
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = P.merge(N[kunci + ["xgb_setel"]], on=kunci, how="inner", validate="one_to_one")


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


print("=" * 100 + "\nD9. GB tersetel lawan GB baku dan rujukan\n" + "=" * 100)
for ylab, ref in B.PASANGAN.items():
    d = B.ambil_sasaran(P, ylab)
    s1 = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
    kol = ["xgb_setel", "xgb", "rf", ref]
    kab = s1.gid.unique()
    blok = {t: B.Blok(g, kol, kab) for t, g in s1.groupby("tag")}

    def fn(b, j):
        y = b.y[j]
        return np.array([B.ap_cepat(y, b.s[m][j]) for m in kol])

    th = np.array([fn(blok[t], slice(None)) for t in B.BIASA])
    Rr = B.dua_tahap(blok, fn, 1000, np.random.default_rng(B.SEED), kab)
    print(f"\n--- {ylab}, rujukan {B.LABEL[ref]} (AP {th[:, 3].mean():.3f}); AP xgb_setel {th[:, 0].mean():.3f}, xgb {th[:, 1].mean():.3f}")
    for a, r in ((0, 3), (1, 3), (0, 1), (0, 2)):
        pt = th[:, a] - th[:, r]; dist = (Rr[:, :, a] - Rr[:, :, r]).mean(1)
        lo, hi = np.percentile(dist, [2.5, 97.5]); tl, thh = t95(pt)
        print(f"  {kol[a] + ' - ' + B.LABEL.get(kol[r], kol[r]):<26}{pt.mean():+.3f}  2tahap [{lo:+.3f}, {hi:+.3f}]  "
              f"t95 [{tl:+.3f}, {thh:+.3f}]  {'LOLOS R16' if lo > 0 and tl > 0 else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt))
    s2 = d[d.skema == "S2"]
    for t in B.EKSTREM:
        x = s2[s2.tag == t]; r0 = B.ap_cepat(x.y, x[ref])
        print(f"  S2 {t}: xgb_setel {B.ap_cepat(x.y, x.xgb_setel) - r0:+.3f}  xgb {B.ap_cepat(x.y, x.xgb) - r0:+.3f}")
print("\nkonfigurasi terpilih:")
print(N.drop_duplicates(["sasaran", "skema", "tag"])[["sasaran", "skema", "tag", "konfigurasi"]].to_string(index=False))
print("\nselesai.")
