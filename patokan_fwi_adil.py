"""
PeatFireBench — baseline FWI yang adil terhadap tempat (panel telaah ketiga, temuan kritis
methods:M1 dan klasternya). Baseline fwi dan fwi_logistik di patokan_prediksi.py menilai FWI pada
satu skala nasional; di sini FWI dibaca relatif terhadap kabupatennya sendiri, dari tahun latih saja.

  fwi_persentil   : persentil fwi_lag1 di antara nilai fwi_lag1 kabupaten itu pada tahun latih
                    (semua bulan) — "apakah bulan lalu luar biasa kering-panas untuk tempat ini?"
  fwi_anomali     : (fwi_lag1 - rata-rata) / simpangan baku kabupaten-bulan-kalender, tahun latih
  klim_fwi        : regresi logistik terskala pada log1p(klimatologi), fwi_persentil, fwi_anomali,
                    suku bulan — FWI sebagai tambahan pada tempat. Klimatologi baris latih dihitung
                    dari data latih itu sendiri, sama dengan baseline klimatologi.
  klim_saja       : regresi logistik yang sama tanpa dua kolom FWI — pembanding yang adil untuk
                    klim_fwi (selisihnya adalah nilai FWI, bukan nilai bentuk model)
  rujukan (bukan prakiraan): fwi_kini_persentil dan klim_fwi_kini, dengan FWI bulan berjalan.

Skema S1 (2019-2025) dan S2 (2015, 2014, 2019), dua sasaran. Keunggulan atas klimatologi
dalam-tahun pada baris yang sama, selang 95 % bootstrap kabupaten (1.000 ulangan, undian sama tiap
tahun), seperti patokan_tabel_naskah.py.

Pakai:    python patokan_fwi_adil.py
Keluaran: layar (DL_FIRE_NASIONAL/penuh/patokan_fwi_adil.log), DL_FIRE_NASIONAL/patokan_fwi_adil.csv
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
df = pd.read_csv(os.path.join(D, "panel_nasional_fitur_penuh.csv"))
df["bulan_lag"] = (df["bulan_ke"] - 2) % 12 + 1          # bulan kalender dari fwi_lag1


def klim(latih, d):
    k = latih.groupby(["gid", "bulan_ke"])["titik_panas"].mean()
    return k.reindex(pd.MultiIndex.from_arrays([d["gid"], d["bulan_ke"]])).fillna(0.0).to_numpy()


def persentil(latih, d, kol):
    ref = {g: np.sort(v.to_numpy()) for g, v in latih.groupby("gid")["fwi_lag1"]}
    out = np.empty(len(d))
    for i, (g, x) in enumerate(zip(d["gid"].to_numpy(), d[kol].to_numpy())):
        r = ref[g]
        out[i] = np.searchsorted(r, x, side="right") / len(r)
    return out


def anomali(latih, d, kol):
    s = latih.groupby(["gid", "bulan_lag"])["fwi_lag1"].agg(["mean", "std"])
    idx = pd.MultiIndex.from_arrays([d["gid"], d["bulan_lag" if kol == "fwi_lag1" else "bulan_ke"]])
    m, sd = s["mean"].reindex(idx).to_numpy(), s["std"].reindex(idx).to_numpy()
    return np.nan_to_num((d[kol].to_numpy() - m) / np.where(sd > 0, sd, np.nan))


def fitur(latih, d, kol):
    return pd.DataFrame({"log_klim": np.log1p(klim(latih, d)), "pers": persentil(latih, d, kol),
                         "anom": anomali(latih, d, kol), "s": d["bulan_sin"].to_numpy(),
                         "c": d["bulan_cos"].to_numpy()})


def logistik(Xl, yl, Xu):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xl, yl)
    return m.predict_proba(Xu)[:, 1]


baris = []
for ylab in ("y_kabupaten", "y_gabungan"):
    lipatan = [("S1", str(t), df[df.tahun < t], df[df.tahun == t]) for t in range(2019, 2026)]
    lipatan += [("S2", str(t), df[df.tahun != t], df[df.tahun == t]) for t in (2015, 2014, 2019)]
    for skema, tag, latih, uji in lipatan:
        Fl, Fu, Fk = fitur(latih, latih, "fwi_lag1"), fitur(latih, uji, "fwi_lag1"), fitur(latih, uji, "fwi_kini")
        yl = latih[ylab].to_numpy()
        b = uji[["gid", "provinsi", "tahun", "bulan", "ada_positif"]].copy()
        b["y"], b["sasaran"], b["skema"], b["tag"] = uji[ylab].to_numpy(), ylab, skema, tag
        b["klimatologi"] = klim(latih, uji)
        b["fwi_persentil"], b["fwi_anomali"] = Fu["pers"].to_numpy(), Fu["anom"].to_numpy()
        b["klim_saja"] = logistik(Fl[["log_klim", "s", "c"]], yl, Fu[["log_klim", "s", "c"]])
        b["klim_fwi"] = logistik(Fl, yl, Fu)
        b["fwi_kini_persentil"] = Fk["pers"].to_numpy()
        b["klim_fwi_kini"] = logistik(Fl, yl, Fk)        # dilatih dengan jeda, dinilai dengan bulan kini
        baris.append(b)
        print(f"  {ylab} {skema} {tag}", flush=True)
P = pd.concat(baris, ignore_index=True)
P.to_csv(os.path.join(D, "patokan_fwi_adil.csv"), index=False)

MODEL = ["fwi_persentil", "fwi_anomali", "klim_saja", "klim_fwi", "fwi_kini_persentil", "klim_fwi_kini"]


def dalam(d, a, b=None):
    return float(np.mean([ap(g.y, g[a]) - (ap(g.y, g[b]) if b else 0) for _, g in d.groupby("tag") if g.y.sum()]))


def boot(d, a, b, n=1000, seed=2026):
    uk = d.gid.unique(); rng = np.random.default_rng(seed)
    per = {t: g for t, g in d.groupby("tag")}
    idx = {t: {k: np.where(g.gid.to_numpy() == k)[0] for k in uk} for t, g in per.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, len(uk)); v = []
        for t, g in per.items():
            j = np.concatenate([idx[t][k] for k in pilih]); y = g.y.to_numpy()[j]
            if y.sum():
                v.append(ap(y, g[a].to_numpy()[j]) - ap(y, g[b].to_numpy()[j]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


for sas in ("y_kabupaten", "y_gabungan"):
    d0 = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    biasa = d0[(d0.skema == "S1") & d0.tag.isin([str(t) for t in range(2020, 2026)])]
    print(f"\n=== {sas}: S1 2020-2025, keunggulan dalam-tahun atas klimatologi [95 %]")
    for m in MODEL:
        lo, hi = boot(biasa, m, "klimatologi")
        print(f"  {m:20s} AUC-PR {dalam(biasa, m):.3f}  vs klim {dalam(biasa, m, 'klimatologi'):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = boot(biasa, "klim_fwi", "klim_saja")
    print(f"  nilai FWI di atas tempat: klim_fwi - klim_saja {dalam(biasa, 'klim_fwi', 'klim_saja'):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = boot(biasa, "klim_fwi_kini", "klim_saja")
    print(f"  rujukan bulan kini:       klim_fwi_kini - klim_saja {dalam(biasa, 'klim_fwi_kini', 'klim_saja'):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"=== {sas}: S2 tahun ekstrem, per tahun")
    for t in ("2015", "2014", "2019"):
        dd = d0[(d0.skema == "S2") & (d0.tag == t)]
        teks = []
        for a, b_ in (("fwi_persentil", "klimatologi"), ("klim_fwi", "klimatologi"), ("klim_fwi", "klim_saja")):
            lo, hi = boot(dd, a, b_, n=300)
            teks.append(f"{a}-{b_} {dalam(dd, a, b_):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {t}: " + "; ".join(teks))
print("\nselesai.")
