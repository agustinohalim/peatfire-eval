"""
PeatFireBench revisi fase D1 dan D2 (Hasil_Patokan_9): kesetaraan (TOST, margin 0,05), selisih
terdeteksi minimum, selang tahun-kecil (t, jackknife tinggal-satu-tahun), himpunan konfirmatori dengan
Holm, bobot kejadian, AUC-PR ternormalisasi prevalensi, dan kesetaraan hit rate kuota 10 %.

Pakai:    python -u patokan_revisi_setara.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_setara.log
"""
import numpy as np
import pandas as pd
from scipy import stats

import patokan_revisi_bantu as B

MARGIN = 0.05
MODEL = ["klimatologi", "klim_frek", "klim_frek1", "persistence", "seasonal_naive", "rasio", "fwi", "fwi_persentil",
         "fwi_logistik", "klim_fwi", "logistik", "log_klim_jeda", "log_ef1_jeda", "rf", "xgb", "mlp"]
KONF = ["rf", "xgb"]
N_ULANG = 1000
B.cek_ap()
P = B.muat_P()
S = pd.read_csv(f"{B.D}/patokan_revisi_sederhana.csv", dtype={"tag": str})
kunci = ["sasaran", "skema", "tag", "gid", "bulan"]
P = P.merge(S[kunci + ["log_ef1_jeda"]], on=kunci, how="left", validate="one_to_one")


def tq(q, df):
    return stats.t.ppf(q, df)


def dua_tahap(blok, fn, n, rng, kab):
    """(n, banyak tahun, k): statistik per tahun terambil, tahun lalu kabupaten."""
    tags = list(blok); out = np.empty((n, len(tags), len(fn(blok[tags[0]], slice(None)))))
    for r in range(n):
        pilih = rng.choice(kab, len(kab))
        tt = [tags[i] for i in rng.integers(0, len(tags), len(tags))]
        cache = {t: fn(blok[t], blok[t].baris(pilih)) for t in set(tt)}
        out[r] = np.array([cache[t] for t in tt])
    return out


def ringkas(nama, pt, dist_rata, w=None, dist_w=None):
    """pt: keunggulan per tahun (6). dist_rata: rata-rata keunggulan per ulangan dua tahap."""
    n = len(pt); m = pt.mean(); sd = pt.std(ddof=1); se = sd / np.sqrt(n)
    t95 = (m - tq(0.975, n - 1) * se, m + tq(0.975, n - 1) * se)
    t90 = (m - tq(0.95, n - 1) * se, m + tq(0.95, n - 1) * se)
    loo = np.array([np.delete(pt, i).mean() for i in range(n)])
    sej = np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())
    j95 = (m - tq(0.975, n - 1) * sej, m + tq(0.975, n - 1) * sej)
    d95 = np.nanpercentile(dist_rata, [2.5, 97.5]); d90 = np.nanpercentile(dist_rata, [5, 95])
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
    setara = (d90[0] > -MARGIN) and (d90[1] < MARGIN) and (t90[0] > -MARGIN) and (t90[1] < MARGIN)
    unggul = d95[0] > 0 and t95[0] > 0
    mdd = (tq(0.975, n - 1) + tq(0.80, n - 1)) * sd
    mdd6 = mdd / np.sqrt(n)
    perlu = next((k for k in range(3, 1001) if (tq(0.975, k - 1) + tq(0.80, k - 1)) * sd / np.sqrt(k) <= m), None) if m > 0 else None
    s = (f"  {nama:<26}{m:+.3f}  2tahap95 [{d95[0]:+.3f}, {d95[1]:+.3f}]  t95 [{t95[0]:+.3f}, {t95[1]:+.3f}]  "
         f"jk95 [{j95[0]:+.3f}, {j95[1]:+.3f}]  2tahap90 [{d90[0]:+.3f}, {d90[1]:+.3f}]  t90 [{t90[0]:+.3f}, {t90[1]:+.3f}]  "
         f"p {p:.3f}  MDD {mdd6:.3f}  thn-perlu {perlu if perlu else '-'}  "
         f"{'UNGGUL' if unggul else ('SETARA' if setara else 'tak-putus')}")
    if w is not None:
        mw = (w * pt).sum() / w.sum()
        s += f"\n  {'':<26}bobot kejadian {mw:+.3f} 2tahap95 [{np.nanpercentile(dist_w, 2.5):+.3f}, {np.nanpercentile(dist_w, 97.5):+.3f}]"
    print(s, flush=True)
    return p


p_konf = {}
print("=" * 100 + f"\nD1/D2. Kesetaraan (margin {MARGIN}), selang tahun-kecil, bobot kejadian (S1 2020-2025)\n" + "=" * 100)
for sas, ref in B.PASANGAN.items():
    d = B.ambil_sasaran(P, sas)
    d = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
    assert d[MODEL].notna().all().all()
    kab = d.gid.unique()
    blok = {t: B.Blok(g, MODEL, kab) for t, g in d.groupby("tag")}
    K = len(MODEL)

    def fn(b, j):
        y = b.y[j]
        return np.r_[[B.ap_cepat(y, b.s[m][j]) for m in MODEL], y.sum(), y.mean()]

    th = np.array([fn(blok[t], slice(None)) for t in B.BIASA])
    rng = np.random.default_rng(B.SEED)
    R = dua_tahap(blok, fn, N_ULANG, rng, kab)
    ir = MODEL.index(ref)
    pos, pi = th[:, K], th[:, K + 1]
    print(f"\n--- {sas}, rujukan {B.LABEL[ref]}: AP rujukan {th[:, ir].mean():.3f}; margin relatif {MARGIN / th[:, ir].mean() * 100:.0f} %; "
          f"{len(kab)} kabupaten")
    print("  prevalensi per tahun " + " ".join(f"{t} {x * 100:.2f}%" for t, x in zip(B.BIASA, pi))
          + "; positif " + " ".join(f"{int(x)}" for x in pos))
    banyak = 0
    for i, m in enumerate(MODEL):
        if i == ir:
            continue
        banyak += 1
        pt = th[:, i] - th[:, ir]
        adv = R[:, :, i] - R[:, :, ir]
        wR = R[:, :, K]
        dist_w = (adv * wR).sum(1) / wR.sum(1)
        p = ringkas(m, pt, adv.mean(1), pos, dist_w)
        if m in KONF:
            p_konf[(sas, m)] = p
            print(f"  {'':<26}per tahun " + " ".join(f"{t} {x:+.3f}" for t, x in zip(B.BIASA, pt)))
        # AUC-PR ternormalisasi prevalensi: (AP_m - AP_r) / (1 - pi)
        ptn = pt / (1 - pi); advn = adv / (1 - R[:, :, K + 1])
        if m in KONF or m == "log_ef1_jeda":
            ringkas(f"  {m} ternormalisasi", ptn, advn.mean(1))
    print(f"  banyak kontras model-rujukan pada sasaran ini: {banyak}")

print("\n--- Himpunan konfirmatori (4 kontras), Holm atas p uji-t tahunan")
urut = sorted(p_konf.items(), key=lambda kv: kv[1])
mmax = 0
for r, (k, p) in enumerate(urut):
    ph = min(1.0, max(mmax, (len(urut) - r) * p)); mmax = ph
    print(f"  {k[0]:<12} {k[1]:<4} p {p:.4f}  p Holm {ph:.4f}  {'lolos' if ph < 0.05 else 'tidak'}")

print("\n" + "=" * 100 + "\nD1. Kesetaraan selisih hit rate kuota 10 % atas EF-1m (per kabupaten, S1)\n" + "=" * 100)


class BlokB(B.Blok):
    def __init__(self, g, kolom, kab):
        super().__init__(g, kolom, kab)
        self.bulan = g.bulan.to_numpy()


def kuota(b, j, m, frac=0.1):
    jj = np.arange(len(b.y))[j]
    bl = b.bulan[jj]; s = b.s[m][jj]; siaga = np.zeros(len(jj), bool)
    for mo in np.unique(bl):
        w = np.where(bl == mo)[0]; n = max(1, int(round(frac * len(w))))
        siaga[w[np.argsort(-s[w], kind="stable")[:n]]] = True
    return siaga


MQ = ["xgb", "rf", "klim_frek1"]
d = B.ambil_sasaran(P, "y_kabupaten"); d = d[(d.skema == "S1") & d.tag.isin(B.BIASA)]
kab = d.gid.unique()
blok = {t: BlokB(g, MQ, kab) for t, g in d.groupby("tag")}


def fq(b, j):
    y = b.y[j]
    return np.array([(kuota(b, j, m) & (y == 1)).sum() / max(y.sum(), 1) for m in MQ])


th = np.array([fq(blok[t], slice(None)) for t in B.BIASA])
R = dua_tahap(blok, fq, N_ULANG, np.random.default_rng(B.SEED), kab)
for m in ("xgb", "rf"):
    i = MQ.index(m)
    ringkas(f"hit {m} - EF-1m", th[:, i] - th[:, 2], (R[:, :, i] - R[:, :, 2]).mean(1))
print("\nselesai.")
