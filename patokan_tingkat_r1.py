"""
Revisi fase R1 VIIRS (Hasil_Patokan_24), dibangun di atas fase P (Hasil_Patokan_22): bagian tingkat (pemecahan fase I4) dengan ambang dari tahun latih (data fase G2).
Bagian data (label ulang per lipatan, kalender, EF-L, RF/GB dilatih ulang) disalin dari
patokan_revisi_ambang_peluang.py tanpa perubahan isi; fungsi geser dan pemecahan disalin dari patokan_tingkat_i.py.

Pakai (satu sasaran per jalan):
    python -u patokan_tingkat_r1.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_r1_kab.log
    python -u patokan_tingkat_r1.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_r1_nas.log
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
PREV = []
for u in range(2020, 2026):
    thn = [t for t in semua if t < u]
    d = label_ulang(df, thn)
    latih, uji = d[d.tahun < u], d[d.tahun == u]
    PREV.append((u, latih[SAS].mean(), df[df.tahun < u][SAS].mean(), uji[SAS].mean(), df[df.tahun == u][SAS].mean()))
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

from scipy.optimize import brentq

print("\nPrevalensi per lipatan (latih ambang tahun latih / latih ambang seluruh panel / uji tahun latih / uji seluruh panel):")
for u, a, b_, c, d_ in PREV:
    print(f"  {u}: {a * 100:.2f}% / {b_ * 100:.2f}% / {c * 100:.2f}% / {d_ * 100:.2f}%")


def logit_i(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def geser(f, ybar):
    lf = logit_i(f)
    h = lambda dl: np.mean(1 / (1 + np.exp(-(lf + dl)))) - ybar
    return 1 / (1 + np.exp(-(lf + brentq(h, -30, 30))))


def mdd80(pt):
    n = len(pt)
    return (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * pt.std(ddof=1) / np.sqrt(n)


PEC = ["xgb", "rf", "kalender"]
g = U.reset_index(drop=True)
for c in PEC:
    g[c + "_pd"] = geser(g[c].to_numpy(), g.y.mean())
    g[c + "_lv"] = np.nan
    for _, x in g.groupby("tag"):
        g.loc[x.index, c + "_lv"] = geser(x[c].to_numpy(), x.y.mean())
SEMUA = ["kalender", "ef1_tingkat", "rf", "xgb"] + [c + w for c in PEC for w in ("_pd", "_lv")]
ix = {c: i for i, c in enumerate(SEMUA)}
kab = g.gid.unique()
blok = {t: B.Blok(x, SEMUA, kab) for t, x in g.groupby("tag")}


def fn(b, j):
    y = b.y[j]
    return np.array([np.mean((b.s[c][j] - y) ** 2) for c in SEMUA])


tags = list(B.BIASA)
th = np.array([fn(blok[t], slice(None)) for t in tags])
R2 = B.dua_tahap(blok, fn, N, np.random.default_rng(B.SEED), kab)


def bss(a, r):
    return 1 - th[:, ix[a]] / th[:, ix[r]], (1 - R2[:, :, ix[a]] / R2[:, :, ix[r]]).mean(1)


print("\n" + "=" * 100 + f"\nP. {SAS}: ambang dari tahun latih, S1 2020-2025 (prevalensi {g.y.mean() * 100:.2f}%)\n" + "=" * 100)
print("Pemeriksaan G2: " + "  ".join(f"BSS {c} atas kalender {bss(c, 'kalender')[0].mean():+.4f}" for c in ("xgb", "rf", "ef1_tingkat")))
print("\nP1 Pemecahan geser-tingkat atas kalender mentah: mentah / offset periode / per tahun; bagian [2 tahap]")
for c in ("xgb", "rf"):
    raw, rd = bss(c, "kalender")
    p_, pdd = bss(c + "_pd", "kalender_pd")
    lv, lvd = bss(c + "_lv", "kalender_lv")
    bp, bt, tot = 1 - pdd / rd, (pdd - lvd) / rd, 1 - lvd / rd
    print(f"    {c:<4} mentah {raw.mean():+.4f}  periode {p_.mean():+.4f}  tahun {lv.mean():+.4f}  "
          f"bagian periode {1 - p_.mean() / raw.mean():.2f} [{np.percentile(bp, 2.5):.2f}, {np.percentile(bp, 97.5):.2f}]  "
          f"bagian tahun {(p_.mean() - lv.mean()) / raw.mean():.2f} [{np.percentile(bt, 2.5):.2f}, {np.percentile(bt, 97.5):.2f}]  "
          f"total {1 - lv.mean() / raw.mean():.2f} [{np.percentile(tot, 2.5):.2f}, {np.percentile(tot, 97.5):.2f}]", flush=True)
    print(f"         per tahun mentah " + " ".join(f"{x:+.3f}" for x in raw) + "  sesudah geser per tahun " +
          " ".join(f"{x:+.3f}" for x in lv))
print("  geser periode (logit) " + "  ".join(
    f"{c} {brentq(lambda dl: np.mean(1 / (1 + np.exp(-(logit_i(g[c].to_numpy()) + dl)))) - g.y.mean(), -30, 30):+.3f}" for c in PEC))
print("\nDeskriptif: MDD80 BSS atas EF-L (enam tahun, df 5), ambang tahun latih")
for c in ("xgb", "rf"):
    pt, _ = bss(c, "ef1_tingkat")
    print(f"    {c:<4} atas EF-L {pt.mean():+.4f}  MDD80 {mdd80(pt):.4f}")


# ---------------------------------------------------------------- fase R (Hasil_Patokan_24)
def logit_r(p, e=1e-6):
    p = min(max(p, e), 1 - e)
    return np.log(p / (1 - p))


print("\n" + "=" * 100 + f"\nR1 VIIRS S1 ({SAS}): L, galat tingkat kalender e_k dan model e_m (logit rata-rata - logit teramati)\n" + "=" * 100)
r1 = []
for i, t in enumerate(tags):
    x = g[g.tag == t]
    raw = 1 - th[i, ix["xgb"]] / th[i, ix["kalender"]]
    lv = 1 - th[i, ix["xgb_lv"]] / th[i, ix["kalender_lv"]]
    ek = logit_r(x.kalender.mean()) - logit_r(x.y.mean())
    em = logit_r(x.xgb.mean()) - logit_r(x.y.mean())
    r1.append((t, raw - lv, ek, em))
    print(f"  {t}: L {raw - lv:+.3f}  e_k {ek:+.3f}  e_m {em:+.3f}  |e_k|-|e_m| {abs(ek) - abs(em):+.3f}", flush=True)
pd.DataFrame(r1, columns=["tahun", "L", "ek", "em"]).assign(rancangan="VIIRS").to_csv(
    f"{B.D}/patokan_r1_viirs_{SAS}.csv", index=False)
print("selesai.")
