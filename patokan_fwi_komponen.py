"""
PeatFireBench — komponen FWI System sebagai baseline adil (telaah editor Natural Hazards, 22 Oktober
2026, X2). patokan_fwi_adil.py menguji FWI saja, karena salinan GFWED di Earth Engine hanya berpita
FWI; di sini Drought Code (DC), Duff Moisture Code (DMC), Fine Fuel Moisture Code (FFMC), Buildup
Index (BUI), dan FWI diambil dari berkas bulanan GFWED NASA (gabung_gfwed_komponen.py, presipitasi
MERRA2.CORRECTED, yang cocok dengan salinan Earth Engine: beda median 0,09 pada September 2015).

Rancangan sama dengan patokan_fwi_adil.py: untuk tiap indeks k, nilai bulan t-1 dibaca relatif
terhadap kabupatennya sendiri dari tahun latih saja (persentil semua bulan dan anomali
kabupaten-bulan-kalender), lalu
  klim_<k>   : regresi logistik terskala pada log1p(klimatologi), persentil, anomali, suku bulan
  klim_saja  : regresi yang sama tanpa indeks (pembanding adil; selisihnya = nilai indeks)
  klim_semua : klimatologi + kelima indeks
Skema S1 (2019-2025) dan S2 (2015, 2014, 2019), dua sasaran. Dalam-tahun AUC-PR, selang 95 %
bootstrap kabupaten (1.000 ulangan tahun biasa, 300 per tahun ekstrem), seperti patokan_fwi_adil.py.

Pakai:    python patokan_fwi_komponen.py   (DL_FIRE_NASIONAL/penuh/patokan_fwi_komponen.log)
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
INDEKS = ["dc", "dmc", "ffmc", "bui", "fwi"]

df = pd.read_csv(os.path.join(D, "panel_nasional_fitur_penuh.csv"))
g = pd.read_csv(os.path.join(D, "gfwed_komponen_MERRA2.CORRECTED.csv"))
g["bulan"] = (pd.PeriodIndex(g.bulan, freq="M") + 1).astype(str)       # nilai bulan t-1 untuk baris t
g = g.rename(columns={"GID_2": "gid", **{k: f"{k}_g1" for k in INDEKS}})
df = df.merge(g[["gid", "bulan"] + [f"{k}_g1" for k in INDEKS]], on=["gid", "bulan"], how="left")
kosong = df[[f"{k}_g1" for k in INDEKS]].isna().any(axis=1)
print(f"baris {len(df)}, tanpa nilai GFWED {kosong.sum()} (diisi median kabupaten-bulan)")
for k in INDEKS:
    c = f"{k}_g1"
    df[c] = df[c].fillna(df.groupby(["gid", "bulan_ke"])[c].transform("median")).fillna(df[c].median())
df["bulan_lag"] = (df["bulan_ke"] - 2) % 12 + 1
print(f"cek: korelasi fwi_g1 dengan fwi_lag1 (Earth Engine) {df.fwi_g1.corr(df.fwi_lag1):.3f}")


def klim(latih, d):
    k = latih.groupby(["gid", "bulan_ke"])["titik_panas"].mean()
    return k.reindex(pd.MultiIndex.from_arrays([d["gid"], d["bulan_ke"]])).fillna(0.0).to_numpy()


def persentil(latih, d, kol):
    ref = {gg: np.sort(v.to_numpy()) for gg, v in latih.groupby("gid")[kol]}
    return np.array([np.searchsorted(ref[gg], x, side="right") / len(ref[gg])
                     for gg, x in zip(d["gid"].to_numpy(), d[kol].to_numpy())])


def anomali(latih, d, kol):
    s = latih.groupby(["gid", "bulan_lag"])[kol].agg(["mean", "std"])
    idx = pd.MultiIndex.from_arrays([d["gid"], d["bulan_lag"]])
    m, sd = s["mean"].reindex(idx).to_numpy(), s["std"].reindex(idx).to_numpy()
    return np.nan_to_num((d[kol].to_numpy() - m) / np.where(sd > 0, sd, np.nan))


def logistik(Xl, yl, Xu):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xl, yl).predict_proba(Xu)[:, 1]


baris = []
for ylab in ("y_kabupaten", "y_gabungan"):
    lipatan = [("S1", str(t), df[df.tahun < t], df[df.tahun == t]) for t in range(2019, 2026)]
    lipatan += [("S2", str(t), df[df.tahun != t], df[df.tahun == t]) for t in (2015, 2014, 2019)]
    for skema, tag, latih, uji in lipatan:
        dasar = lambda d: pd.DataFrame({"log_klim": np.log1p(klim(latih, d)), "s": d["bulan_sin"].to_numpy(),
                                         "c": d["bulan_cos"].to_numpy()})
        Fl, Fu = dasar(latih), dasar(uji)
        for k in INDEKS:
            c = f"{k}_g1"
            Fl[f"p_{k}"], Fu[f"p_{k}"] = persentil(latih, latih, c), persentil(latih, uji, c)
            Fl[f"a_{k}"], Fu[f"a_{k}"] = anomali(latih, latih, c), anomali(latih, uji, c)
        yl = latih[ylab].to_numpy()
        b = uji[["gid", "provinsi", "tahun", "bulan", "ada_positif"]].copy()
        b["y"], b["sasaran"], b["skema"], b["tag"] = uji[ylab].to_numpy(), ylab, skema, tag
        b["klimatologi"] = klim(latih, uji)
        b["klim_saja"] = logistik(Fl[["log_klim", "s", "c"]], yl, Fu[["log_klim", "s", "c"]])
        for k in INDEKS:
            kol = ["log_klim", "s", "c", f"p_{k}", f"a_{k}"]
            b[f"klim_{k}"] = logistik(Fl[kol], yl, Fu[kol])
        b["klim_semua"] = logistik(Fl, yl, Fu)
        baris.append(b)
        print(f"  {ylab} {skema} {tag}", flush=True)
P = pd.concat(baris, ignore_index=True)
P.to_csv(os.path.join(D, "patokan_fwi_komponen.csv"), index=False)
MODEL = [f"klim_{k}" for k in INDEKS] + ["klim_semua"]


def dalam(d, a, b=None):
    return float(np.mean([ap(gg.y, gg[a]) - (ap(gg.y, gg[b]) if b else 0) for _, gg in d.groupby("tag") if gg.y.sum()]))


def boot(d, a, b, n=1000, seed=2026):
    uk = d.gid.unique(); rng = np.random.default_rng(seed)
    per = {t: gg for t, gg in d.groupby("tag")}
    idx = {t: {k: np.where(gg.gid.to_numpy() == k)[0] for k in uk} for t, gg in per.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, len(uk)); v = []
        for t, gg in per.items():
            j = np.concatenate([idx[t][k] for k in pilih]); y = gg.y.to_numpy()[j]
            if y.sum():
                v.append(ap(y, gg[a].to_numpy()[j]) - ap(y, gg[b].to_numpy()[j]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


for sas in ("y_kabupaten", "y_gabungan"):
    d0 = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    biasa = d0[(d0.skema == "S1") & d0.tag.isin([str(t) for t in range(2020, 2026)])]
    print(f"\n=== {sas}: S1 2020-2025, nilai indeks di atas tempat (klim_<k> - klim_saja), dalam-tahun AUC-PR [95 %]")
    print(f"  klim_saja AUC-PR {dalam(biasa, 'klim_saja'):.3f}, klimatologi {dalam(biasa, 'klimatologi'):.3f}")
    for m in MODEL:
        lo, hi = boot(biasa, m, "klim_saja")
        print(f"  {m:11s} AUC-PR {dalam(biasa, m):.3f}  - klim_saja {dalam(biasa, m, 'klim_saja'):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"=== {sas}: S2 tahun ekstrem, klim_<k> - klim_saja per tahun")
    for t in ("2015", "2014", "2019"):
        dd = d0[(d0.skema == "S2") & (d0.tag == t)]
        teks = []
        for m in MODEL:
            lo, hi = boot(dd, m, "klim_saja", n=300)
            teks.append(f"{m[5:]} {dalam(dd, m, 'klim_saja'):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {t}: " + "; ".join(teks))
print("\nselesai.")
