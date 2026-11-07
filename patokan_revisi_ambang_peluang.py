"""
PeatFireBench revisi fase G2 (Hasil_Patokan_12): skor peluang dengan ambang dari tahun latih.

Label diulang per lipatan dengan `label_ulang` (disalin dari patokan_revisi_ambang.py: persentil 90 kabupaten,
batas bawah 10, dan persentil 90 gabungan, dari tahun latih). RF dan GB dari patokan_prediksi_kepekaan_penuh.csv
(dilatih ulang pada label ulang, benih 42). Kalender EF-1m / EF-3m dari R.klim_frek pada label ulang tahun latih;
EF-L dengan fungsi fase E/F (disalin dari patokan_revisi_tingkat.py) pada label ulang; anomali nasional tidak
berubah. Baris per kabupaten: kabupaten dengan positif di tahun latih lipatan. S1 2020-2025.

Pakai (satu sasaran per jalan):
    python -u patokan_revisi_ambang_peluang.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_revisi_ambang_peluang_kab.log
    python -u patokan_revisi_ambang_peluang.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_revisi_ambang_peluang_nas.log
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
KAPPA = 2.0
N = 1000
BATAS_BAWAH = 10
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


# ------------------------------------------------- EF-L (disalin dari patokan_revisi_tingkat.py)
GID = np.sort(df.gid.unique()); THN = np.arange(df.tahun.min(), df.tahun.max() + 1)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
G_, T_, M_ = df.gid.map(gi).to_numpy(), df.tahun.map(ti).to_numpy(), (df.bulan_ke - 1).to_numpy()
NAS = pd.Series(df.tp_lag1.to_numpy()).groupby([T_, M_]).sum()
LN = np.full((len(THN), 12), np.nan)
LN[NAS.index.get_level_values(0), NAS.index.get_level_values(1)] = np.log1p(NAS.to_numpy())


def jendela(a):
    return a + np.roll(a, 1, axis=-1) + np.roll(a, -1, axis=-1)


def rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    Yl = Y[:, msk, :]
    S1 = np.nansum(Yl, 1); N1 = np.sum(~np.isnan(Yl), 1)
    S3, N3 = jendela(S1), jendela(N1)
    s1, n1, s3, n3 = S1[g, m], N1[g, m].astype(float), S3[g, m], N3[g, m].astype(float)
    if keluar_sendiri:
        yo = Y[g, t, m]; ada = ~np.isnan(yo); yo0 = np.where(ada, yo, 0.0)
        s1 = s1 - yo0; n1 = n1 - ada
        w = np.stack([Y[g, t, (m - 1) % 12], yo, Y[g, t, (m + 1) % 12]])
        s3 = s3 - np.nansum(w, 0); n3 = n3 - np.sum(~np.isnan(w), 0)
    bagi = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), 0.0)
    ef3 = bagi(s3, n3)
    return {"ef1_halus": (s1 + KAPPA * ef3) / (n1 + KAPPA)}


def anomali(tahun_latih, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(LN[msk], 0); n = np.sum(~np.isnan(LN[msk]), 0).astype(float)
    s, n = s[m], n[m]
    if keluar_sendiri:
        s = s - LN[t, m]; n = n - 1
    return LN[t, m] - s / np.maximum(n, 1)


def logit(p, e=1e-4):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def indeks(Y, tahun):
    g, t, m = np.where(~np.isnan(Y[:, np.isin(THN, tahun), :]))
    t = np.flatnonzero(np.isin(THN, tahun))[t]
    return g, t, m


def efl_uji(Y, uji):
    tahun_latih = [x for x in THN if x < uji]
    g, t, m = indeks(Y, tahun_latih)
    rl = rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=True)
    X = np.column_stack([logit(rl["ef1_halus"]), anomali(tahun_latih, t, m, keluar_sendiri=True)])
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(X, Y[g, t, m])
    g, t, m = indeks(Y, [uji])
    ru = rujukan(Y, tahun_latih, g, t, m)
    p = lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), anomali(tahun_latih, t, m)]))[:, 1]
    return pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[a]}-{b + 1:02d}" for a, b in zip(t, m)], "ef1_tingkat": p})


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


B.cek_ap()
baris = []
for u in range(2020, 2026):
    thn = [t for t in semua if t < u]
    d = label_ulang(df, thn)
    latih, uji = d[d.tahun < u], d[d.tahun == u]
    b = uji[["gid", "bulan"]].copy()
    b["y_ulang"] = uji[SAS].to_numpy()
    b["kalender"] = R.klim_frek(latih, uji, SAS, 0 if SAS == "y_kabupaten" else 1)
    pos = latih.groupby("gid")[SAS].sum()
    b["ada_lipatan"] = b.gid.map(pos).fillna(0).gt(0).astype(int).to_numpy()
    Y = np.full((len(GID), len(THN), 12), np.nan)
    Y[G_, T_, M_] = d[SAS].to_numpy(float)
    if SAS == "y_kabupaten":
        Y[~np.isin(GID, pos[pos > 0].index)] = np.nan
    b = b.merge(efl_uji(Y, u), on=["gid", "bulan"], how="left")
    b["skema"], b["tag"] = "S1", str(u)
    baris.append(b)
    print(f"S1 {u}: ambang dari {thn[0]}-{thn[-1]}", flush=True)
U = pd.concat(baris, ignore_index=True)
K = pd.read_csv(f"{B.D}/patokan_prediksi_kepekaan_penuh.csv", dtype={"tag": str})
K = K[(K.sasaran == SAS) & (K.skema == "S1")][["tag", "gid", "bulan", "y", "rf", "xgb"]]
U = U.merge(K, on=["tag", "gid", "bulan"], how="inner", validate="one_to_one")
print(f"baris {len(U)}; y kepekaan = y dihitung ulang: {(U.y == U.y_ulang).mean():.4f}")
if SAS == "y_kabupaten":
    U = U[U.ada_lipatan == 1]
    print(f"baris per kabupaten (positif di tahun latih): {len(U)}; EF-L kosong {int(U.ef1_tingkat.isna().sum())}")
U = U.dropna(subset=["ef1_tingkat"])

KOL = ["kalender", "ef1_tingkat", "rf", "xgb"]
kab = U.gid.unique()
blok = {t: B.Blok(g, KOL, kab) for t, g in U.groupby("tag")}


def fn(b, j):
    y = b.y[j]
    return np.array([np.mean((b.s[c][j] - y) ** 2) for c in KOL] + [B.ap_cepat(y, b.s[c][j]) for c in KOL])


th = np.array([fn(blok[t], slice(None)) for t in B.BIASA])
R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)
ix = {c: i for i, c in enumerate(KOL)}
print("\n" + "=" * 100 + f"\nG2. {SAS}: ambang dari tahun latih, S1 2020-2025 (prevalensi {U.y.mean() * 100:.2f}%)\n" + "=" * 100)
print("  Brier rata-rata tahun: " + "  ".join(f"{c} {th[:, ix[c]].mean():.5f}" for c in KOL))
print("  AUC-PR rata-rata tahun: " + "  ".join(f"{c} {th[:, 4 + ix[c]].mean():.3f}" for c in KOL))
for a, r in (("rf", "kalender"), ("xgb", "kalender"), ("ef1_tingkat", "kalender"), ("rf", "ef1_tingkat"), ("xgb", "ef1_tingkat")):
    i, k = ix[a], ix[r]
    pt = 1 - th[:, i] / th[:, k]; dist = (1 - R2[:, :, i] / R2[:, :, k]).mean(1)
    lo, hi = np.percentile(dist, [2.5, 97.5]); tl, tu = t95(pt)
    print(f"    BSS {a:<12} atas {r:<12}{pt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
          f"{'  LOLOS' if lo > 0 and tl > 0 else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in pt), flush=True)
print("selesai.")
