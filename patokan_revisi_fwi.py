"""
PeatFireBench revisi fase B4 — ISI, tampilan gabungan antartahun, dan DC September 2015 (Hasil_Patokan_7).

1. ISI bulanan per kabupaten dari berkas GFWED yang sama (agregasi gabung_gfwed_komponen.py apa adanya)
   -> DL_FIRE_NASIONAL/gfwed_isi_MERRA2.CORRECTED.csv; cek FWI hasilnya = gfwed_komponen_MERRA2.CORRECTED.csv.
2. klim_isi dengan rancangan patokan_fwi_komponen.py (log klimatologi, musim, persentil dan anomali
   kabupaten dari tahun latih); klim_saja dihitung ulang dan dicek sama dengan berkas lama.
3. Dalam-tahun (seperti Hasil_Patokan_5) untuk ISI; gabungan antartahun S1 2020-2025 untuk DC, DMC,
   FFMC, BUI, FWI, kelimanya, dan ISI, selisih terhadap klim_saja [95 %, bootstrap kabupaten 1.000].
4. DC September 2015 lawan September 2016-2025, Kalimantan Tengah dan Sumatera Selatan.

Pakai:    python patokan_revisi_fwi.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_fwi.log
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gabung_gfwed_komponen as G

warnings.filterwarnings("ignore")
D = G.D
SEED = 2026
BIASA = [str(t) for t in range(2020, 2026)]
PRES = "MERRA2.CORRECTED"
ISI_CSV = os.path.join(D, f"gfwed_isi_{PRES}.csv")

# 1. ISI
if not os.path.exists(ISI_CSV):
    G.KOMP = ["ISI", "FWI"]
    G.agregasi(PRES).to_csv(ISI_CSV, index=False)
isi = pd.read_csv(ISI_CSV)
lama = pd.read_csv(os.path.join(D, f"gfwed_komponen_{PRES}.csv"))
c = isi.merge(lama[["GID_2", "bulan", "fwi"]], on=["GID_2", "bulan"], suffixes=("", "_lama"))
print(f"ISI: {len(isi)} baris; cek FWI lawan berkas komponen: selisih maks {np.nanmax(np.abs(c.fwi - c.fwi_lama)):.2e}")

# 2. rancangan patokan_fwi_komponen.py
df = pd.read_csv(os.path.join(D, "panel_nasional_fitur_penuh.csv"))
g = isi.copy()
g["bulan"] = (pd.PeriodIndex(g.bulan, freq="M") + 1).astype(str)
g = g.rename(columns={"GID_2": "gid", "isi": "isi_g1"})
df = df.merge(g[["gid", "bulan", "isi_g1"]], on=["gid", "bulan"], how="left")
print(f"baris tanpa ISI: {df.isi_g1.isna().sum()} (diisi median kabupaten-bulan)")
df["isi_g1"] = df.isi_g1.fillna(df.groupby(["gid", "bulan_ke"]).isi_g1.transform("median")).fillna(df.isi_g1.median())
df["bulan_lag"] = (df["bulan_ke"] - 2) % 12 + 1


def klim(latih, d):
    k = latih.groupby(["gid", "bulan_ke"])["titik_panas"].mean()
    return k.reindex(pd.MultiIndex.from_arrays([d["gid"], d["bulan_ke"]])).fillna(0.0).to_numpy()


def persentil(latih, d, kol):
    ref = {gg: np.sort(v.to_numpy()) for gg, v in latih.groupby("gid")[kol]}
    return np.array([np.searchsorted(ref[gg], x, side="right") / len(ref[gg]) for gg, x in zip(d["gid"].to_numpy(), d[kol].to_numpy())])


def anomali(latih, d, kol):
    s = latih.groupby(["gid", "bulan_lag"])[kol].agg(["mean", "std"])
    idx = pd.MultiIndex.from_arrays([d["gid"], d["bulan_lag"]])
    m, sd = s["mean"].reindex(idx).to_numpy(), s["std"].reindex(idx).to_numpy()
    return np.nan_to_num((d[kol].to_numpy() - m) / np.where(sd > 0, sd, np.nan))


def logistik(Xl, yl, Xu):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xl, yl).predict_proba(Xu)[:, 1]


baris = []
for ylab in ("y_kabupaten", "y_gabungan"):
    lip = [("S1", str(t), df[df.tahun < t], df[df.tahun == t]) for t in range(2019, 2026)]
    lip += [("S2", str(t), df[df.tahun != t], df[df.tahun == t]) for t in (2015, 2014, 2019)]
    for sk, tag, lt, uj in lip:
        F = {}
        for nm, d in (("l", lt), ("u", uj)):
            F[nm] = pd.DataFrame({"log_klim": np.log1p(klim(lt, d)), "s": d.bulan_sin.to_numpy(), "c": d.bulan_cos.to_numpy(),
                                  "p_isi": persentil(lt, d, "isi_g1"), "a_isi": anomali(lt, d, "isi_g1")})
        yl = lt[ylab].to_numpy()
        b = uj[["gid", "bulan"]].copy(); b["sasaran"], b["skema"], b["tag"] = ylab, sk, tag
        b["klim_saja_ulang"] = logistik(F["l"][["log_klim", "s", "c"]], yl, F["u"][["log_klim", "s", "c"]])
        b["klim_isi"] = logistik(F["l"], yl, F["u"])
        baris.append(b)
    print(f"  {ylab} selesai", flush=True)
B = pd.concat(baris, ignore_index=True)
K = pd.read_csv(os.path.join(D, "patokan_fwi_komponen.csv"), dtype={"tag": str})
K = K.merge(B, on=["sasaran", "skema", "tag", "gid", "bulan"], how="inner", validate="one_to_one")
print(f"cek klim_saja dihitung ulang lawan berkas lama: selisih maks {np.abs(K.klim_saja - K.klim_saja_ulang).max():.2e}")


def dalam(d, a, b):
    return float(np.mean([ap(x.y, x[a]) - ap(x.y, x[b]) for _, x in d.groupby("tag") if x.y.sum()]))


def gabung(d, a, b):
    return ap(d.y, d[a]) - ap(d.y, d[b])


def boot(d, a, b, f, n):
    uk = d.gid.unique(); rng = np.random.default_rng(SEED)
    idx = {k: np.where(d.gid.to_numpy() == k)[0] for k in uk}
    out = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(uk, len(uk))])
        out.append(f(d.iloc[j], a, b))
    return np.percentile(out, [2.5, 97.5])


def fmt(x, lo, hi):
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


MODEL = ["klim_dc", "klim_dmc", "klim_ffmc", "klim_bui", "klim_fwi", "klim_semua", "klim_isi"]
for sas in ("y_kabupaten", "y_gabungan"):
    d0 = K[K.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    b = d0[(d0.skema == "S1") & d0.tag.isin(BIASA)]
    print(f"\n=== {sas}: S1 2020-2025, selisih terhadap klim_saja [95 %, 1.000]")
    print(f"  klim_saja: dalam-tahun {np.mean([ap(x.y, x.klim_saja) for _, x in b.groupby('tag')]):.3f}, gabungan {ap(b.y, b.klim_saja):.3f}")
    for m in MODEL:
        wy = fmt(dalam(b, m, "klim_saja"), *boot(b, m, "klim_saja", dalam, 1000)) if m == "klim_isi" else f"{dalam(b, m, 'klim_saja'):+.3f} (Hasil_Patokan_5)"
        print(f"  {m:<11} dalam-tahun {wy:<34} gabungan {fmt(gabung(b, m, 'klim_saja'), *boot(b, m, 'klim_saja', gabung, 1000))}")
    print("  S2, klim_isi - klim_saja dalam tahun itu [95 %, 300]:")
    for t in ("2015", "2014", "2019"):
        dd = d0[(d0.skema == "S2") & (d0.tag == t)]
        print(f"    {t}: {fmt(gabung(dd, 'klim_isi', 'klim_saja'), *boot(dd, 'klim_isi', 'klim_saja', gabung, 300))}")

# 4. DC September 2015
prov = pd.read_csv(os.path.join(D, "panel_nasional_fitur_penuh.csv"), usecols=["gid", "provinsi"]).drop_duplicates()
L = lama.merge(prov, left_on="GID_2", right_on="gid")
print("\nDC bulanan GFWED, rerata kabupaten:")
for p in ("KalimantanTengah", "SumateraSelatan"):
    x = L[L.provinsi == p]
    s15 = x[x.bulan == "2015-09"].dc.mean()
    lain = x[x.bulan.isin([f"{t}-09" for t in range(2016, 2026)])].groupby("bulan").dc.mean()
    print(f"  {p}: September 2015 {s15:.0f}; September 2016-2025 rerata {lain.mean():.0f}, maks {lain.max():.0f} ({lain.idxmax()})")
print("\nselesai.")
