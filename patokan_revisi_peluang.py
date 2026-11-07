"""
PeatFireBench revisi fase E (Hasil_Patokan_10): rujukan peluang yang sepadan.

E1  rujukan peluang dari tahun latih lipatan: konstan, konstan_bulan, ef1, ef3, ef1_pasca (tahun >= 2016),
    ef1_halus ((s + 2*ef3)/(n + 2)); rujukan_peluang = calon dengan Brier terendah dalam validasi
    tinggalkan-satu-tahun di tahun latih; ef1_tingkat = logistik logit(ef1_halus) + anomali api nasional
    bulan lalu (untuk baris latih tanpa tahunnya sendiri).
E2  pemecahan Brier oracle: tiap ramalan digeser di skala logit agar rata-ratanya = laju teramati tahun itu.
E3  BSS semua kolom bernilai [0, 1] atas EF pasangan, rujukan_peluang, ef1_tingkat.
E4  log score pada eps 1e-4 dan 1e-3; bagian positif dengan EF-1m < 1e-3.

Pakai:    python -u patokan_revisi_peluang.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_peluang.log
Keluaran: DL_FIRE_NASIONAL/patokan_revisi_peluang.csv (tidak masuk repo)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

import patokan_revisi_bantu as B

warnings.filterwarnings("ignore")
N_ULANG = 1000
KAPPA = 2.0
CALON = ["konstan", "konstan_bulan", "ef1", "ef3", "ef1_pasca", "ef1_halus"]
KUNCI = ["sasaran", "skema", "tag", "gid", "bulan"]
# satu sasaran per jalan (memori laptop sempit): python patokan_revisi_peluang.py y_kabupaten
SASARAN = [sys.argv[1]] if len(sys.argv) > 1 else list(B.PASANGAN)
PASANGAN = {k: v for k, v in B.PASANGAN.items() if k in SASARAN}

# ---------------------------------------------------------------- panel sebagai larik (gid, tahun, bulan)
df = pd.read_csv(os.path.join(B.D, "panel_nasional_fitur_penuh.csv"),
                 usecols=["gid", "tahun", "bulan", "bulan_ke", "tp_lag1", "y_kabupaten", "y_gabungan", "ada_positif"])
GID = np.sort(df.gid.unique()); THN = np.arange(df.tahun.min(), df.tahun.max() + 1)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
df["_g"] = df.gid.map(gi); df["_t"] = df.tahun.map(ti); df["_m"] = df.bulan_ke - 1
ADA = df.groupby("_g").ada_positif.max().reindex(range(len(GID))).fillna(0).to_numpy() == 1
NAS = df.groupby(["_t", "_m"]).tp_lag1.sum()
LN = np.full((len(THN), 12), np.nan)
LN[NAS.index.get_level_values(0), NAS.index.get_level_values(1)] = np.log1p(NAS.to_numpy())


def larik(ylab):
    Y = np.full((len(GID), len(THN), 12), np.nan)
    Y[df._g, df._t, df._m] = df[ylab].to_numpy(float)
    if ylab == "y_kabupaten":
        Y[~ADA] = np.nan                       # populasi evaluasi per kabupaten
    return Y


def jendela(a):
    return a + np.roll(a, 1, axis=-1) + np.roll(a, -1, axis=-1)


def rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=False):
    """rujukan untuk baris (g, t, m) dari tahun latih; keluar_sendiri: buang tahun t baris itu (baris latih)."""
    msk = np.isin(THN, tahun_latih)
    Yl = Y[:, msk, :]
    S1 = np.nansum(Yl, 1); N1 = np.sum(~np.isnan(Yl), 1)
    S3, N3 = jendela(S1), jendela(N1)
    pas = msk & (THN >= 2016)
    Sp = np.nansum(Y[:, pas, :], 1); Np = np.sum(~np.isnan(Y[:, pas, :]), 1)
    Sm = np.nansum(Yl, (0, 1)); Nm = np.sum(~np.isnan(Yl), (0, 1))
    s1, n1, s3, n3 = S1[g, m], N1[g, m].astype(float), S3[g, m], N3[g, m].astype(float)
    sp, np_ = Sp[g, m], Np[g, m].astype(float)
    sm, nm = np.full(len(g), Sm.sum()), np.full(len(g), float(Nm.sum()))
    smb, nmb = Sm[m], Nm[m].astype(float)
    if keluar_sendiri:
        yo = Y[g, t, m]; ada = ~np.isnan(yo); yo0 = np.where(ada, yo, 0.0)
        s1 = s1 - yo0; n1 = n1 - ada
        w = np.stack([Y[g, t, (m - 1) % 12], yo, Y[g, t, (m + 1) % 12]])
        s3 = s3 - np.nansum(w, 0); n3 = n3 - np.sum(~np.isnan(w), 0)
        dlm_pasca = THN[t] >= 2016
        sp = sp - np.where(dlm_pasca, yo0, 0); np_ = np_ - np.where(dlm_pasca, ada, 0)
        sm = sm - yo0; nm = nm - ada; smb = smb - yo0; nmb = nmb - ada
    bagi = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), 0.0)
    out = {"konstan": bagi(sm, nm), "konstan_bulan": bagi(smb, nmb), "ef1": bagi(s1, n1), "ef3": bagi(s3, n3)}
    out["ef1_pasca"] = np.where(np_ > 0, bagi(sp, np_), out["konstan_bulan"])
    out["ef1_halus"] = (s1 + KAPPA * out["ef3"]) / (n1 + KAPPA)
    return out


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


def bs(y, p):
    return np.mean((p - y) ** 2)


def lipatan(Y, ylab, skema, uji):
    tahun_latih = [x for x in THN if (x < uji if skema == "S1" else x != uji)]
    # pemilihan rujukan peluang: tinggalkan satu tahun latih
    br = {c: [] for c in CALON}
    for v in tahun_latih:
        g, t, m = indeks(Y, [v])
        r = rujukan(Y, [x for x in tahun_latih if x != v], g, t, m)
        y = Y[g, t, m]
        for c in CALON:
            br[c].append(bs(y, r[c]))
    br = {c: np.mean(v) for c, v in br.items()}
    pilih = min(br, key=br.get)
    # ef1_tingkat
    g, t, m = indeks(Y, tahun_latih)
    rl = rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=True)
    X = np.column_stack([logit(rl["ef1_halus"]), anomali(tahun_latih, t, m, keluar_sendiri=True)])
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(X, Y[g, t, m])
    # baris uji
    g, t, m = indeks(Y, [uji])
    ru = rujukan(Y, tahun_latih, g, t, m)
    ru["rujukan_peluang"] = ru[pilih]
    ru["ef1_tingkat"] = lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), anomali(tahun_latih, t, m)]))[:, 1]
    bingkai = pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[a]}-{b + 1:02d}" for a, b in zip(t, m)], **ru})
    bingkai["sasaran"], bingkai["skema"], bingkai["tag"] = ylab, skema, str(uji)
    info = f"{ylab:<12}{skema} {uji}: pilih {pilih:<14}" + " ".join(f"{c} {br[c]:.5f}" for c in CALON) \
        + f"   ef1_tingkat koef ef1 {lg.coef_[0, 0]:+.3f} anomali {lg.coef_[0, 1]:+.3f}"
    return bingkai, info, pilih


print("=" * 110 + "\nE1. Rujukan peluang per lipatan (Brier tinggalkan-satu-tahun di tahun latih)\n" + "=" * 110, flush=True)
bagian, PILIH = [], {}
for ylab in PASANGAN:
    Y = larik(ylab)
    for skema, ujis in (("S1", range(2020, 2026)), ("S2", (2015, 2014, 2019))):
        for u in ujis:
            b, info, pilih = lipatan(Y, ylab, skema, u)
            bagian.append(b); PILIH[(ylab, skema, str(u))] = pilih
            print("  " + info, flush=True)
R = pd.concat(bagian, ignore_index=True)
R.to_csv(os.path.join(B.D, f"patokan_revisi_peluang_{'_'.join(SASARAN)}.csv"), index=False)

# ---------------------------------------------------------------- gabung semua prediksi
P = B.muat_P(fwi=False)
P = P[P.sasaran.isin(SASARAN)]
for f, kol in (("patokan_revisi_sederhana.csv", None), ("patokan_revisi_fdrs.csv", None), ("patokan_revisi_setel.csv", ["xgb_setel"]),
               ("patokan_fwi_adil.csv", ["fwi_persentil", "fwi_anomali", "klim_saja", "klim_fwi", "fwi_kini_persentil", "klim_fwi_kini"])):
    X = pd.read_csv(os.path.join(B.D, f), dtype={"tag": str})
    X = X[X.sasaran.isin(SASARAN)]
    X = X[KUNCI + (kol or [c for c in X.columns if c not in KUNCI])]
    P = P.merge(X, on=KUNCI, how="left", validate="one_to_one")
P = P.merge(R, on=KUNCI, how="left", validate="one_to_one")
for s_ in SASARAN:
    d0 = B.ambil_sasaran(P, s_)
    d0 = d0[d0.ef1.notna()]
    print(f"\ncek {s_}: ef1 dihitung ulang vs klim_frek1 tersimpan: selisih maks {np.abs(d0.ef1 - d0.klim_frek1).max():.2e}; "
          f"ef3 vs klim_frek {np.abs(d0.ef3 - d0.klim_frek).max():.2e}; baris dicek {len(d0)}")

BUKAN = set(KUNCI) | {"provinsi", "pulau", "tahun", "ada_positif", "y", "konfigurasi"}
KOL = []
PE = pd.concat([B.ambil_sasaran(P, s) for s in SASARAN])
PE = PE[((PE.skema == "S1") & PE.tag.isin(B.BIASA)) | ((PE.skema == "S2") & PE.tag.isin(B.EKSTREM))]
for c in P.columns:
    if c in BUKAN or c in CALON + ["rujukan_peluang", "ef1_tingkat"]:
        continue
    v = PE[c]
    if not np.issubdtype(v.dtype, np.number):
        print(f"  E3 keluar: {c} (bukan angka)"); continue
    if v.isna().any():
        print(f"  E3 keluar: {c} ({v.isna().sum()} baris kosong)"); continue
    if v.min() < 0 or v.max() > 1 + 1e-5:   # klim_frek: pemecah seri 1e-6 di atas 1
        print(f"  E3 keluar: {c} (nilai di luar [0,1]: {v.min():.3g}..{v.max():.3g})"); continue
    KOL.append(c)
print(f"E3 kolom ({len(KOL)}): {', '.join(KOL)}", flush=True)


def t95(pt):
    n = len(pt); se = pt.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return pt.mean() - q * se, pt.mean() + q * se


def geser(f, ybar):
    lf = logit(f, 1e-6)
    h = lambda dlt: np.mean(1 / (1 + np.exp(-(lf + dlt)))) - ybar
    return 1 / (1 + np.exp(-(lf + brentq(h, -30, 30))))


def baris_bss(nama, th, R2, i, r, kel, tags):
    bt = 1 - th[:, i] / th[:, r]
    if kel == "biasa":
        lo, hi = np.percentile((1 - R2[:, :, i] / R2[:, :, r]).mean(1), [2.5, 97.5]); tl, tu = t95(bt)
        lolos = lo > 0 and tl > 0
        print(f"    {nama:<44}{bt.mean():+.4f} 2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]"
              f"{'  LOLOS' if lolos else ''}  per tahun " + " ".join(f"{x:+.3f}" for x in bt), flush=True)
        return bt.mean(), lolos
    print(f"    {nama:<44}" + "  ".join(f"{t} {x:+.3f}" for t, x in zip(tags, bt)), flush=True)
    return bt.mean(), None


RINGKAS = []
for sas, ref in PASANGAN.items():
    d = B.ambil_sasaran(P, sas)
    for kel, sk, tags in (("biasa", "S1", B.BIASA), ("ekstrem", "S2", B.EKSTREM)):
        g = d[(d.skema == sk) & d.tag.isin(tags)].copy()
        print("\n" + "=" * 110 + f"\n{sas}, {kel} (rujukan pasangan {B.LABEL[ref]}); rujukan peluang terpilih: "
              + ", ".join(f"{t} {PILIH[(sas, sk, t)]}" for t in tags) + "\n" + "=" * 110, flush=True)
        # E2: ramalan tingkat-benar per tahun
        UTAMA = ["xgb", "rf", "mlp", "log_ef1_jeda", "xgb_setel"]
        ACU = [ref, "rujukan_peluang", "ef1_tingkat"]
        for c in UTAMA + ACU:
            g[c + "_lv"] = np.nan
            for _, x in g.groupby("tag"):
                g.loc[x.index, c + "_lv"] = geser(x[c].to_numpy(), x.y.mean())
        kol = KOL + CALON + ["rujukan_peluang", "ef1_tingkat"] + [c + "_lv" for c in UTAMA + ACU]
        kol = list(dict.fromkeys(kol))
        ix = {c: i for i, c in enumerate(kol)}
        kab = g.gid.unique()
        blok = {t: B.Blok(x, kol, kab) for t, x in g.groupby("tag")}

        def fn(b, j):
            y = b.y[j]
            return np.array([bs(y, b.s[c][j]) for c in kol])

        th = np.array([fn(blok[t], slice(None)) for t in tags])
        R2 = B.dua_tahap(blok, fn, N_ULANG, np.random.default_rng(B.SEED), kab) if kel == "biasa" else None
        print("  Brier rata-rata tahun: " + "  ".join(f"{c} {th[:, ix[c]].mean():.5f}" for c in
                                                       CALON + ["rujukan_peluang", "ef1_tingkat", "xgb", "rf"]))
        print(f"\n  R24. Calon konstan atas {B.LABEL[ref]}")
        for c in ("konstan", "konstan_bulan", "ef1_pasca", "ef1_halus", "rujukan_peluang", "ef1_tingkat"):
            baris_bss(f"BSS {c} atas {B.LABEL[ref]}", th, R2, ix[c], ix[ref], kel, tags)
        print(f"\n  E3. BSS semua kolom peluang atas {B.LABEL[ref]} / rujukan_peluang / ef1_tingkat")
        for acu in (ref, "rujukan_peluang", "ef1_tingkat"):
            print(f"  -- atas {acu}")
            n_lolos = n_pos = 0
            for c in KOL:
                if c == acu:
                    continue
                v, lolos = baris_bss(f"{c}", th, R2, ix[c], ix[acu], kel, tags)
                n_pos += v > 0; n_lolos += bool(lolos)
                RINGKAS.append((sas, kel, acu, c, v, lolos))
            print(f"     ringkas: titik > 0 pada {n_pos} dari {len(KOL) - (acu in KOL)}; lolos kedua selang {n_lolos}")
        print("\n  E2. Pemecahan oracle tingkat tahun (BSS mentah / tingkat-benar / bagian tingkat)")
        for acu in (ref, "rujukan_peluang"):
            for c in UTAMA:
                i, r, il, rl = ix[c], ix[acu], ix[c + "_lv"], ix[acu + "_lv"]
                mentah = 1 - th[:, i] / th[:, r]; lv = 1 - th[:, il] / th[:, rl]
                if kel == "biasa":
                    a = (1 - R2[:, :, i] / R2[:, :, r]).mean(1); b2 = (1 - R2[:, :, il] / R2[:, :, rl]).mean(1)
                    bag = 1 - b2 / a
                    lo, hi = np.percentile(b2, [2.5, 97.5]); tl, tu = t95(lv)
                    blo, bhi = np.percentile(bag, [2.5, 97.5])
                    print(f"    {c:<13} atas {acu:<16} mentah {mentah.mean():+.4f}  tingkat-benar {lv.mean():+.4f} "
                          f"2tahap [{lo:+.4f}, {hi:+.4f}] t95 [{tl:+.4f}, {tu:+.4f}]  bagian tingkat "
                          f"{1 - lv.mean() / mentah.mean():.2f} [{blo:.2f}, {bhi:.2f}]  per tahun tb "
                          + " ".join(f"{x:+.3f}" for x in lv), flush=True)
                else:
                    print(f"    {c:<13} atas {acu:<16} mentah " + " ".join(f"{t} {x:+.3f}" for t, x in zip(tags, mentah))
                          + "   tingkat-benar " + " ".join(f"{t} {x:+.3f}" for t, x in zip(tags, lv)), flush=True)
        if True:
            print("\n  E4. Log score: eps 1e-4 / 1e-3")
            yy = g.y.to_numpy()
            print(f"    bagian positif dengan {B.LABEL[ref]} < 1e-3: {(g.loc[g.y == 1, ref] < 1e-3).mean():.3f}; "
                  f"rujukan_peluang < 1e-3: {(g.loc[g.y == 1, 'rujukan_peluang'] < 1e-3).mean():.3f}")
            for eps in (1e-4, 1e-3):
                def ll(y, p):
                    p = np.clip(p, eps, 1 - eps)
                    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                for acu in (ref, "rujukan_peluang", "ef1_tingkat"):
                    dl = []
                    for c in ("xgb", "rf"):
                        per = np.array([ll(x.y.to_numpy(), x[acu].to_numpy()) - ll(x.y.to_numpy(), x[c].to_numpy())
                                        for _, x in g.groupby("tag")])
                        dl.append(f"{c} {per.mean():+.4f}" + (" t95 [{:+.4f}, {:+.4f}]".format(*t95(per)) if kel == "biasa" else ""))
                    print(f"    eps {eps:g} dlog atas {acu:<16}" + "   ".join(dl), flush=True)

print("\n" + "=" * 110 + "\nRingkasan R27 (tahun biasa): kolom dengan BSS lolos kedua selang / titik > 0\n" + "=" * 110)
Rk = pd.DataFrame(RINGKAS, columns=["sasaran", "kel", "acuan", "kolom", "bss", "lolos"])
for (s, a), x in Rk[Rk.kel == "biasa"].groupby(["sasaran", "acuan"], sort=False):
    print(f"  {s:<12} atas {a:<16} lolos {int(x.lolos.sum())} dari {len(x)}; titik > 0 {int((x.bss > 0).sum())}; "
          f"tidak lolos: {', '.join(x.loc[~x.lolos.astype(bool), 'kolom'])}")
print("\nselesai.")
