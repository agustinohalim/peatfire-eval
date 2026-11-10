"""
Revisi fase S (Hasil_Patokan_25): semua prakiraan Tabel 2, Tabel 4 dan Tabel 6 (kolom tahun biasa) dengan ambang dari
tahun latih, S1 2020-2025, satu sasaran per jalan. Skrip ini hanya membangun prakiraan; angka dihitung oleh
patokan_ambang_s_tabel.py.

Label. Per lipatan (tahun uji u), `label_ulang` (disalin tanpa perubahan dari patokan_revisi_ambang.py: persentil 90
kabupaten dengan batas bawah 10, persentil 90 gabungan) dari tahun panel < u. Label ini dipakai untuk baris latih dan
uji lipatan itu: melatih semua model, menghitung semua rujukan, dan (Tabel 6) memilih ambang siaga di tahun latih.
Per kabupaten: hanya kabupaten yang punya positif di tahun latih lipatan (seperti G2, kolom ada_lipatan); kabupaten
lain juga dikeluarkan dari larik rujukan EF (seperti patokan_revisi_ambang_peluang.py).

Prakiraan (nama kolom sama dengan kode asal; sumber dan baris yang disalin ditandai di setiap blok):
  tanpa label, digabung dari berkas tersimpan (S1, kunci tag-gid-bulan):
    klimatologi, persistence, seasonal_naive, rasio, fwi           patokan_prediksi_penuh.csv
    fwi_persentil                                                   patokan_fwi_adil.csv
    dc_nasional, dc_persentil, ffmc_nasional                        patokan_revisi_fdrs.csv
  dilatih ulang pada label ulang, sudah tersimpan (benih 42):
    rf, xgb, mlp, logistik, fwi_logistik                            patokan_prediksi_kepekaan_penuh.csv
  dihitung di sini pada label ulang:
    klim_frek1 (EF-1m), klim_frek (EF-3m), log_klim_jeda            patokan_revisi_prediksi.py (R.baseline_baru)
    log_ef1_jeda, xgb_sejarah                                        patokan_revisi_sederhana.py
    log_ef1_fdrs (+ ef1_api, ef1_cuaca, ef1_keduanya, xgb_tanpa_api) patokan_revisi_fdrs.py
    klim_fwi                                                         patokan_fwi_adil.py
    xgb_setel (grid 18, validasi dua tahun latih terbaru)            patokan_revisi_setel.py
    konstan, rujukan_peluang (EF-P), ef1_tingkat (EF-L)              patokan_revisi_peluang.py
    efd (EF-D), efld (EF-LD)                                         patokan_tingkat_o.py (= patokan_tingkat_m.py)
    xgb_nas                                                          patokan_revisi_tingkat.py
    xgb_nasdc                                                        patokan_tingkat_o.py
  baris tahun latih (untuk ambang Tabel 6 dan pilihan rujukan C3):
    xgb, klimatologi, klim_frek, klim_frek1 luar-lipatan per tahun latih  patokan_revisi_oof.py (xgb hanya per kabupaten)
    ef1_tingkat baris latih (nilai pas, tinggalkan-tahun-sendiri)          patokan_revisi_tingkat.py (efl_lipatan)

Satu-satunya perubahan terhadap kode asal: label (latih dan uji) dan kolom turunan label (y bulan lalu di
sederhana/fdrs, himpunan kabupaten per kabupaten di validasi xgb_setel dan di larik EF) dihitung dari label ulang
lipatan. Fungsi yang namanya bertabrakan antarsumber dibungkus dalam fungsi `bagian_*` tanpa mengubah isinya.
Skrip ini tidak mencetak metrik; hanya kemajuan dan banyak baris.

Pakai (satu sasaran per jalan, memori):
    python -u patokan_ambang_s.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_ambang_s_kab.log
    python -u patokan_ambang_s.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_ambang_s_nas.log
Keluaran: DL_FIRE_NASIONAL/patokan_ambang_s_<sasaran>.csv (baris uji) dan
          DL_FIRE_NASIONAL/patokan_ambang_s_<sasaran>_latih.csv (baris tahun latih); tidak masuk repo.
"""
import os
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import patokan_revisi_bantu as B
import patokan_revisi_prediksi as R  # fungsi saja; bagian utama dijaga __main__
from kelompok_fitur import KELOMPOK

warnings.filterwarnings("ignore")
SAS = sys.argv[1]
assert SAS in B.PASANGAN, SAS
UJI = list(range(2020, 2026))
KAPPA = 2.0
BATAS_BAWAH = 10
CALON = ["konstan", "konstan_bulan", "ef1", "ef3", "ef1_pasca", "ef1_halus"]
OUT = os.path.join(B.D, f"patokan_ambang_s_{SAS}.csv")
OUT_LATIH = os.path.join(B.D, f"patokan_ambang_s_{SAS}_latih.csv")
T0 = time.time()

df = R.df
# ---- disalin dari patokan_revisi_ambang.py:20-31 tanpa perubahan isi
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
# ---- akhir salinan


# ---- panel sebagai larik: disalin dari patokan_tingkat_o.py:39-56 (indeks, LN, DC) tanpa perubahan isi; df di sini
# R.df (berkas sama, panel_nasional_fitur_penuh.csv, urutan sama), kolom _g/_t/_m ditambahkan ke R.df.
GID = np.sort(df.gid.unique()); THN = np.arange(df.tahun.min(), df.tahun.max() + 1)
gi = {g: i for i, g in enumerate(GID)}; ti = {t: j for j, t in enumerate(THN)}
df["_g"] = df.gid.map(gi); df["_t"] = df.tahun.map(ti); df["_m"] = df.bulan_ke - 1
NAS = df.groupby(["_t", "_m"]).tp_lag1.sum()
LN = np.full((len(THN), 12), np.nan)
LN[NAS.index.get_level_values(0), NAS.index.get_level_values(1)] = np.log1p(NAS.to_numpy())

C = pd.read_csv(os.path.join(B.D, "gfwed_komponen_MERRA2.CORRECTED.csv"), usecols=["GID_2", "bulan", "dc"])
C = C.rename(columns={"GID_2": "gid"}).sort_values(["gid", "bulan"])
C["dc_lag1"] = C.groupby("gid").dc.shift(1)
x = df[["gid", "bulan", "_g", "_t", "_m"]].merge(C[["gid", "bulan", "dc_lag1"]], on=["gid", "bulan"], how="left",
                                                 validate="one_to_one")
print(f"DC bulan lalu hilang {x.dc_lag1.isna().mean() * 100:.2f}% (diisi median kabupaten)", flush=True)
x["dc_lag1"] = x.dc_lag1.fillna(x.groupby("gid").dc_lag1.transform("median")).fillna(x.dc_lag1.median())
DC = np.full((len(GID), len(THN), 12), np.nan)
DC[x._g, x._t, x._m] = np.log1p(x.dc_lag1.to_numpy())
del x, C
# ---- akhir salinan
# ---- disalin dari patokan_revisi_ambang_peluang.py:48 (indeks larik untuk Y per lipatan)
G_, T_, M_ = df.gid.map(gi).to_numpy(), df.tahun.map(ti).to_numpy(), (df.bulan_ke - 1).to_numpy()
# ---- disalin dari patokan_fwi_adil.py:37 (kolom tanpa label, ditambahkan ke R.df)
df["bulan_lag"] = (df["bulan_ke"] - 2) % 12 + 1          # bulan kalender dari fwi_lag1


# ---- fungsi EF: disalin dari patokan_revisi_peluang.py:56-138 tanpa perubahan isi (jendela, rujukan, anomali,
# logit, indeks, bs, lipatan); ef1_halus dan anomali_dc dari patokan_tingkat_o.py:71-83 dan 95-100; efl_lipatan dari
# patokan_revisi_tingkat.py:102-115. rujukan di revisi_tingkat hanya mengembalikan ef1_halus dengan rumus yang sama.
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


def ef1_halus(Y, tahun_latih, g, t, m, keluar_sendiri=False):
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
    ef3 = np.where(n3 > 0, s3 / np.maximum(n3, 1), 0.0)
    return (s1 + KAPPA * ef3) / (n1 + KAPPA)


def anomali_dc(tahun_latih, g, t, m, keluar_sendiri=False):
    msk = np.isin(THN, tahun_latih)
    s = np.nansum(DC[:, msk, :], 1)[g, m]; n = np.sum(~np.isnan(DC[:, msk, :]), 1)[g, m].astype(float)
    if keluar_sendiri:
        s = s - DC[g, t, m]; n = n - 1
    return DC[g, t, m] - s / np.maximum(n, 1)


def efl_lipatan(Y, skema, uji):
    """EF-L untuk baris latih (nilai pas, kalender dan anomali tinggalkan-tahun-sendiri) dan baris uji."""
    tahun_latih = [x for x in THN if (x < uji if skema == "S1" else x != uji)]
    g, t, m = indeks(Y, tahun_latih)
    rl = rujukan(Y, tahun_latih, g, t, m, keluar_sendiri=True)
    X = np.column_stack([logit(rl["ef1_halus"]), anomali(tahun_latih, t, m, keluar_sendiri=True)])
    lg = LogisticRegression(C=1e6, max_iter=2000).fit(X, Y[g, t, m])
    latih = pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[a]}-{b + 1:02d}" for a, b in zip(t, m)],
                          "tahun_dalam": THN[t], "y": Y[g, t, m], "ef1_tingkat": lg.predict_proba(X)[:, 1]})
    g, t, m = indeks(Y, [uji])
    ru = rujukan(Y, tahun_latih, g, t, m)
    p = lg.predict_proba(np.column_stack([logit(ru["ef1_halus"]), anomali(tahun_latih, t, m)]))[:, 1]
    uji_df = pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[a]}-{b + 1:02d}" for a, b in zip(t, m)], "ef1_tingkat_ulang": p})
    return latih, uji_df
# ---- akhir salinan fungsi EF


def efd_lipatan(Y, u):
    """EF-D dan EF-LD satu lipatan; isi loop disalin dari patokan_tingkat_o.py:117-129 tanpa perubahan isi."""
    tl = [v for v in THN if v < u]
    g, t, m = indeks(Y, tl)
    h, a, d = logit(ef1_halus(Y, tl, g, t, m, True)), anomali(tl, t, m, True), anomali_dc(tl, g, t, m, True)
    yl = Y[g, t, m]
    fit = lambda *X: LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack(X), yl)
    mL, mD, mLD = fit(h, a), fit(h, d), fit(h, a, d)
    g, t, m = indeks(Y, [u])
    h, a, d = logit(ef1_halus(Y, tl, g, t, m)), anomali(tl, t, m), anomali_dc(tl, g, t, m)
    return pd.DataFrame({"gid": GID[g], "bulan": [f"{THN[p]}-{q + 1:02d}" for p, q in zip(t, m)], "tag": str(u),
                         "efl_ulang": mL.predict_proba(np.column_stack([h, a]))[:, 1],
                         "efd": mD.predict_proba(np.column_stack([h, d]))[:, 1],
                         "efld": mLD.predict_proba(np.column_stack([h, a, d]))[:, 1]})


def xgb_nas_lipatan(D, sk, u):
    """GB + anomali nasional; isi loop disalin dari patokan_revisi_tingkat.py:147-160 tanpa perubahan isi.
    D = R.df berlabel ulang (kolom _t/_m sudah ada, sama dengan revisi_tingkat.py:146)."""
    FIT = R.FITUR + ["anomali_nasional"]
    tl = [x for x in THN if (x < u if sk == "S1" else x != u)]
    latih = D[D.tahun.isin(tl)].copy(); uji = D[D.tahun == u].copy()
    latih["anomali_nasional"] = anomali(tl, latih._t.to_numpy(), latih._m.to_numpy(), keluar_sendiri=True)
    uji["anomali_nasional"] = anomali(tl, uji._t.to_numpy(), uji._m.to_numpy())
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(latih[FIT].fillna(0.0).to_numpy(), latih[SAS].to_numpy())
    b = uji[["gid", "bulan"]].copy()
    b["xgb_nas"] = xg.predict_proba(uji[FIT].fillna(0.0).to_numpy())[:, 1]
    return b


def xgb_nasdc_lipatan(Dp, u):
    """GB + anomali nasional + anomali DC; isi loop disalin dari patokan_tingkat_o.py:142-156 tanpa perubahan isi.
    Dp = R.df berlabel ulang (kolom _g/_t/_m sudah ada, sama dengan tingkat_o.py:140)."""
    FIT = R.FITUR + ["anomali_nasional", "anomali_dc"]
    tl = [v for v in THN if v < u]
    latih = Dp[Dp.tahun.isin(tl)].copy(); uji = Dp[Dp.tahun == u].copy()
    for x, ks in ((latih, True), (uji, False)):
        gg, tt, mm = x._g.to_numpy(int), x._t.to_numpy(int), x._m.to_numpy(int)
        x["anomali_nasional"] = anomali(tl, tt, mm, keluar_sendiri=ks)
        x["anomali_dc"] = anomali_dc(tl, gg, tt, mm, keluar_sendiri=ks)
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
    xg.fit(latih[FIT].fillna(0.0).to_numpy(), latih[SAS].to_numpy())
    b = uji[["gid", "bulan"]].copy()
    b["xgb_nasdc"] = xg.predict_proba(uji[FIT].fillna(0.0).to_numpy())[:, 1]
    return b


def jeda_label(d):
    """y bulan lalu dari label ulang lipatan: disalin dari patokan_revisi_sederhana.py:28-29 (= patokan_revisi_fdrs.py:40-41),
    dijalankan per lipatan sesudah label_ulang, bukan sekali pada label baku."""
    for ylab in B.PASANGAN:
        d[ylab + "_lag1"] = d.groupby("gid")[ylab].shift(1).fillna(0).to_numpy()
    return d


def bagian_sederhana():
    # ---- disalin dari patokan_revisi_sederhana.py:26-27 dan 33-60 tanpa perubahan isi
    SEJARAH = ["tp_lag1", "tp_lag2", "tp_lag3", "tp_lag12", "bulan_sin", "bulan_cos"]
    df = R.df.sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)

    def ef1_tanpa_tahun_sendiri(latih, ylab):
        tab = R.tabel_klim(latih, ylab)
        s, n = R.ambil(tab, latih, 0)
        x = latih[ylab].to_numpy()
        f = np.where(n > 1, (s - x) / np.maximum(n - 1, 1), 0.0)
        k = R.klim_tanpa_tahun_sendiri(latih)
        return f + 1e-6 * k / (1 + k)

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def X_ef1(d, ef1, ylab):
        return np.column_stack([logit(ef1), d[ylab + "_lag1"].to_numpy()]
                               + [np.log1p(d[c].fillna(0.0).to_numpy()) for c in ("tp_lag1", "tp_lag2", "tp_lag3")]
                               + [d.bulan_sin.to_numpy(), d.bulan_cos.to_numpy()])

    def baru(latih, uji, ylab):
        lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        lg.fit(X_ef1(latih, ef1_tanpa_tahun_sendiri(latih, ylab), ylab), latih[ylab].to_numpy())
        p_lg = lg.predict_proba(X_ef1(uji, R.klim_frek(latih, uji, ylab, 0), ylab))[:, 1]
        xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                           reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        xg.fit(latih[SEJARAH].fillna(0.0).to_numpy(), latih[ylab].to_numpy())
        return {"log_ef1_jeda": p_lg, "xgb_sejarah": xg.predict_proba(uji[SEJARAH].fillna(0.0).to_numpy())[:, 1]}
    # ---- akhir salinan
    return SimpleNamespace(df=df, baru=baru)


def bagian_fdrs():
    # ---- disalin dari patokan_revisi_fdrs.py:32-39, 42, 45-101 tanpa perubahan isi (baris 40-41, y bulan lalu,
    # dipindah ke jeda_label per lipatan)
    df = R.df.sort_values(["gid", "tahun", "bulan_ke"]).reset_index(drop=True)
    C = pd.read_csv(f"{B.D}/gfwed_komponen_MERRA2.CORRECTED.csv", usecols=["GID_2", "bulan", "dc", "ffmc"])
    C = C.rename(columns={"GID_2": "gid"}).sort_values(["gid", "bulan"])
    C["dc_lag1"] = C.groupby("gid").dc.shift(1); C["ffmc_lag1"] = C.groupby("gid").ffmc.shift(1)
    df = df.merge(C[["gid", "bulan", "dc_lag1", "ffmc_lag1"]], on=["gid", "bulan"], how="left", validate="one_to_one")
    print(f"baris panel {len(df)}; DC bulan lalu hilang {df.dc_lag1.isna().mean() * 100:.2f}% (diisi median kabupaten)", flush=True)
    for c in ("dc_lag1", "ffmc_lag1"):
        df[c] = df[c].fillna(df.groupby("gid")[c].transform("median")).fillna(df[c].median())
    TANPA_API = [f for f in R.FITUR if f not in KELOMPOK["api"]]

    def ef1_tanpa_tahun_sendiri(latih, ylab):
        tab = R.tabel_klim(latih, ylab)
        s, n = R.ambil(tab, latih, 0)
        x = latih[ylab].to_numpy()
        f = np.where(n > 1, (s - x) / np.maximum(n - 1, 1), 0.0)
        k = R.klim_tanpa_tahun_sendiri(latih)
        return f + 1e-6 * k / (1 + k)

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def kol(d, nama, ylab):
        if nama == "api":
            return [d[ylab + "_lag1"].to_numpy(), np.log1p(d.tp_lag1.fillna(0.0).to_numpy())]
        if nama == "cuaca":
            return [d.dc_lag1.to_numpy(), d.ffmc_lag1.to_numpy(), d.hujan_lag1.fillna(0.0).to_numpy(),
                    d.hujan_anom_lag1.fillna(0.0).to_numpy()]
        if nama == "fdrs":
            return [d.dc_lag1.to_numpy(), d.ffmc_lag1.to_numpy()]

    def X(d, ef1, ylab, kelompok):
        c = [logit(ef1), d.bulan_sin.to_numpy(), d.bulan_cos.to_numpy()]
        for k in kelompok:
            c += kol(d, k, ylab)
        return np.column_stack(c)

    SPEK = {"log_ef1_fdrs": ["fdrs"], "ef1_api": ["api"], "ef1_cuaca": ["cuaca"], "ef1_keduanya": ["api", "cuaca"]}

    def persentil(latih, uji, c):
        out = np.zeros(len(uji))
        ref = {g: np.sort(x[c].to_numpy()) for g, x in latih.groupby("gid")}
        for g, idx in uji.groupby("gid").groups.items():
            r = ref.get(g)
            v = uji.loc[idx, c].to_numpy()
            out[uji.index.get_indexer(idx)] = np.searchsorted(r, v, side="right") / len(r) if r is not None else 0.5
        return out

    def baru(latih, uji, ylab):
        out = {"dc_nasional": uji.dc_lag1.to_numpy(), "ffmc_nasional": uji.ffmc_lag1.to_numpy(),
               "dc_persentil": persentil(latih, uji, "dc_lag1")}
        ef_l = ef1_tanpa_tahun_sendiri(latih, ylab); ef_u = R.klim_frek(latih, uji, ylab, 0)
        for nama, kel in SPEK.items():
            lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            lg.fit(X(latih, ef_l, ylab, kel), latih[ylab].to_numpy())
            out[nama] = lg.predict_proba(X(uji, ef_u, ylab, kel))[:, 1]
        xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                           reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
        xg.fit(latih[TANPA_API].fillna(0.0).to_numpy(), latih[ylab].to_numpy())
        out["xgb_tanpa_api"] = xg.predict_proba(uji[TANPA_API].fillna(0.0).to_numpy())[:, 1]
        return out
    # ---- akhir salinan
    return SimpleNamespace(df=df, baru=baru)


def bagian_fwi_adil():
    # ---- disalin dari patokan_fwi_adil.py:40-69 tanpa perubahan isi
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
    # ---- akhir salinan
    return SimpleNamespace(fitur=fitur, logistik=logistik)


def bagian_setel():
    # ---- disalin dari patokan_revisi_setel.py:142 dan 146-158 tanpa perubahan isi. ap_val memakai kolom ada_positif;
    # di sini kolom itu diisi per lipatan dengan "kabupaten punya positif di tahun latih lipatan" (label ulang).
    DEPTH, NEST, MCW = (2, 3, 5), (150, 300, 600), (1, 10)

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
    # ---- akhir salinan
    return SimpleNamespace(DEPTH=DEPTH, NEST=NEST, MCW=MCW, model=model, ap_val=ap_val)


def setel_lipatan(d, u):
    """isi loop disalin dari patokan_revisi_setel.py:165-182 untuk satu lipatan S1 dan satu sasaran, tanpa perubahan isi."""
    latih = d[d.tahun < u]
    uji = d[d.tahun == u]
    val_th = sorted(latih.tahun.unique())[-2:]
    dalam, val = latih[~latih.tahun.isin(val_th)], latih[latih.tahun.isin(val_th)]
    skor = {}
    for dp in ST.DEPTH:
        for mc in ST.MCW:
            m = ST.model(dp, mc).fit(dalam[R.FITUR].fillna(0.0).to_numpy(), dalam[SAS].to_numpy())
            for n in ST.NEST:
                p = m.predict_proba(val[R.FITUR].fillna(0.0).to_numpy(), iteration_range=(0, n))[:, 1]
                skor[(dp, n, mc)] = ST.ap_val(val, p, SAS)
    best = max(skor, key=skor.get)
    m = ST.model(best[0], best[2], best[1]).fit(latih[R.FITUR].fillna(0.0).to_numpy(), latih[SAS].to_numpy())
    b = uji[["gid", "bulan"]].copy()
    b["xgb_setel"] = m.predict_proba(uji[R.FITUR].fillna(0.0).to_numpy())[:, 1]
    b["konfigurasi"] = f"d{best[0]}_n{best[1]}_mcw{best[2]}"
    return b


def oof_lipatan(d, u):
    """prakiraan luar-lipatan pada tahun latih; isi loop disalin dari patokan_revisi_oof.py:27-34. xgb (R.dipelajari_xgb)
    hanya untuk sasaran per kabupaten (Tabel 6); log_klim_jeda tidak dihitung (tidak dipakai Tabel 6 atau C3)."""
    latih_luar = d[d.tahun < u]
    out = []
    for v in sorted(latih_luar.tahun.unique()):
        latih, uji = latih_luar[latih_luar.tahun != v], latih_luar[latih_luar.tahun == v]
        b = uji[["gid", "bulan"]].copy()
        b["y"] = uji[SAS].to_numpy()
        b["klimatologi"] = R.klim_rata(latih, uji)
        b["klim_frek"] = R.klim_frek(latih, uji, SAS, 1); b["klim_frek1"] = R.klim_frek(latih, uji, SAS, 0)
        if SAS == "y_kabupaten":
            b["xgb"] = R.dipelajari_xgb(latih, uji, SAS)
        b["tahun_dalam"] = v
        out.append(b)
    return pd.concat(out, ignore_index=True)


SD = bagian_sederhana()
FD = bagian_fdrs()
FA = bagian_fwi_adil()
ST = bagian_setel()
KG = ["gid", "bulan"]


def gabung(b, x, kol):
    return b.merge(x[KG + kol], on=KG, how="left", validate="one_to_one")


print(f"{SAS}: rujukan pasangan {B.LABEL[B.PASANGAN[SAS]]}; panel {len(df):,} baris, tahun {THN[0]}-{THN[-1]}", flush=True)
BARIS_UJI, BARIS_LATIH = [], []
for u in UJI:
    thn = [t for t in semua if t < u]
    d = label_ulang(df, thn)
    latih, uji = d[d.tahun < u], d[d.tahun == u]
    pos = latih.groupby("gid")[SAS].sum()
    b = uji[["gid", "bulan"]].copy()
    b["tag"] = str(u)
    b["y"] = uji[SAS].to_numpy()
    b["ada_lipatan"] = b.gid.map(pos).fillna(0).gt(0).astype(int).to_numpy()

    # EF-1m, EF-3m, riwayat lokal MC (patokan_revisi_prediksi.py:91-96, dipanggil tanpa perubahan)
    for k, v in R.baseline_baru(latih, uji, SAS).items():
        b[k] = v
    # klimatologi + FWI (patokan_fwi_adil.py:77-84, hanya klim_fwi)
    Fl, Fu = FA.fitur(latih, latih, "fwi_lag1"), FA.fitur(latih, uji, "fwi_lag1")
    b["klim_fwi"] = FA.logistik(Fl, latih[SAS].to_numpy(), Fu)
    print(f"  {u}: ambang dari {thn[0]}-{thn[-1]}; kalender, riwayat lokal MC, klimatologi + FWI ({time.time() - T0:.0f} dtk)",
          flush=True)

    # riwayat lokal EF-1m, GB riwayat api (patokan_revisi_sederhana.py:66-72)
    ds = jeda_label(label_ulang(SD.df, thn))
    ls, us = ds[ds.tahun < u], ds[ds.tahun == u]
    x = us[KG].copy()
    for k, v in SD.baru(ls, us, SAS).items():
        x[k] = v
    b = gabung(b, x, ["log_ef1_jeda", "xgb_sejarah"])
    # riwayat lokal EF-1m + DC/FFMC dan varian jeda (patokan_revisi_fdrs.py:108-114); kolom tanpa label dari berkas
    dd = jeda_label(label_ulang(FD.df, thn))
    lf, uf = dd[dd.tahun < u], dd[dd.tahun == u].reset_index(drop=True)
    x = uf[KG].copy()
    for k, v in FD.baru(lf, uf, SAS).items():
        x[k] = v
    b = gabung(b, x, ["log_ef1_fdrs", "ef1_api", "ef1_cuaca", "ef1_keduanya", "xgb_tanpa_api"])
    print(f"  {u}: riwayat lokal EF-1m, GB riwayat api, FDRS ({time.time() - T0:.0f} dtk)", flush=True)

    # GB tersetel (patokan_revisi_setel.py), validasi per kabupaten pada kabupaten positif di tahun latih lipatan
    dset = d.copy()
    dset["ada_positif"] = dset.gid.map(pos).fillna(0).gt(0).astype(int).to_numpy()
    b = gabung(b, setel_lipatan(dset, u), ["xgb_setel", "konfigurasi"])
    print(f"  {u}: GB tersetel {b.konfigurasi.iloc[0]} ({time.time() - T0:.0f} dtk)", flush=True)

    # larik label ulang untuk EF (patokan_revisi_ambang_peluang.py:122-125)
    Y = np.full((len(GID), len(THN), 12), np.nan)
    Y[G_, T_, M_] = d[SAS].to_numpy(float)
    if SAS == "y_kabupaten":
        Y[~np.isin(GID, pos[pos > 0].index)] = np.nan
    # EF-P, laju konstan, EF-L (patokan_revisi_peluang.py:112-138)
    bk, _, pilih = lipatan(Y, SAS, "S1", u)
    bk["rujukan_peluang_calon"] = pilih
    b = gabung(b, bk, ["konstan", "rujukan_peluang", "rujukan_peluang_calon", "ef1_tingkat"])
    # EF-L baris latih (patokan_revisi_tingkat.py:102-115); baris uji harus sama dengan lipatan()
    lt, uj = efl_lipatan(Y, "S1", u)
    c = b.merge(uj, on=KG, how="inner", validate="one_to_one")
    assert len(c) == b.ef1_tingkat.notna().sum() and np.abs(c.ef1_tingkat - c.ef1_tingkat_ulang).max() < 1e-8
    lt["sumber"], lt["tag"] = "efl", str(u)
    BARIS_LATIH.append(lt)
    # EF-D, EF-LD (patokan_tingkat_o.py:117-129); EF-L dihitung ulang di sana harus sama
    b = gabung(b, efd_lipatan(Y, u), ["efl_ulang", "efd", "efld"])
    c = b[b.ef1_tingkat.notna()]
    assert np.abs(c.efl_ulang - c.ef1_tingkat).max() < 1e-4
    b = b.drop(columns=["efl_ulang"])
    print(f"  {u}: EF-P ({pilih}), laju konstan, EF-L, EF-D, EF-LD ({time.time() - T0:.0f} dtk)", flush=True)

    # GB + nasional (patokan_revisi_tingkat.py:143-162), GB + nasional + DC (patokan_tingkat_o.py:136-160)
    b = gabung(b, xgb_nas_lipatan(d, "S1", u), ["xgb_nas"])
    b = gabung(b, xgb_nasdc_lipatan(d, u), ["xgb_nasdc"])
    print(f"  {u}: GB + nasional, GB + nasional + DC ({time.time() - T0:.0f} dtk)", flush=True)

    # baris tahun latih luar-lipatan (patokan_revisi_oof.py)
    o = oof_lipatan(d, u)
    if SAS == "y_kabupaten":
        o = o[o.gid.map(pos).fillna(0).gt(0).to_numpy()]
    o["sumber"], o["tag"] = "oof", str(u)
    BARIS_LATIH.append(o)
    print(f"  {u}: luar-lipatan tahun latih {o.tahun_dalam.nunique()} tahun, {len(o):,} baris; "
          f"EF-L latih {len(lt):,} baris ({time.time() - T0:.0f} dtk)", flush=True)
    BARIS_UJI.append(b)

U = pd.concat(BARIS_UJI, ignore_index=True)
print(f"baris uji (semua kabupaten) {len(U):,}", flush=True)

# ---------------------------------------------------------------- prakiraan tersimpan
KT = ["tag", "gid", "bulan"]


def tersimpan(berkas, kol, sasaran=SAS):
    X = pd.read_csv(os.path.join(B.D, berkas), dtype={"tag": str})
    X = X[(X.sasaran == sasaran) & (X.skema == "S1") & X.tag.isin(B.BIASA)]
    return X[KT + kol]


# dilatih ulang pada label ulang (patokan_kepekaan_ambang.py, benih 42)
K = tersimpan("patokan_prediksi_kepekaan_penuh.csv", ["y", "rf", "xgb", "mlp", "logistik", "fwi_logistik"])
U = U.merge(K.rename(columns={"y": "y_kepekaan"}), on=KT, how="inner", validate="one_to_one")
assert (U.y == U.y_kepekaan).all(), "label ulang berbeda dengan patokan_prediksi_kepekaan_penuh.csv"
U = U.drop(columns=["y_kepekaan"])
# tanpa label (sama untuk kedua ambang)
for berkas, kol in (("patokan_prediksi_penuh.csv", ["klimatologi", "persistence", "seasonal_naive", "rasio", "fwi"]),
                    ("patokan_fwi_adil.csv", ["fwi_persentil"]),
                    ("patokan_revisi_fdrs.csv", ["dc_nasional", "dc_persentil", "ffmc_nasional"])):
    U = U.merge(tersimpan(berkas, kol), on=KT, how="left", validate="one_to_one")
print(f"baris uji sesudah gabung berkas tersimpan {len(U):,}", flush=True)

if SAS == "y_kabupaten":
    U = U[U.ada_lipatan == 1].reset_index(drop=True)
PRAKIRAAN = [c for c in U.columns if c not in ("gid", "bulan", "tag", "y", "ada_lipatan", "konfigurasi",
                                                    "rujukan_peluang_calon")]
assert not U[PRAKIRAAN].isna().any().any(), U[PRAKIRAAN].isna().sum()[lambda s: s > 0]
U.to_csv(OUT, index=False)
print(f"tersimpan: {OUT}  baris {len(U):,}, kabupaten {U.gid.nunique()}, kolom prakiraan {len(PRAKIRAAN)}")
for t, x in U.groupby("tag"):
    print(f"  {t}: {len(x):,} baris, {x.gid.nunique()} kabupaten")

L = pd.concat(BARIS_LATIH, ignore_index=True)
L = L[["sumber", "tag", "tahun_dalam", "gid", "bulan", "y"] + [c for c in ("klimatologi", "klim_frek", "klim_frek1", "xgb",
                                                                            "ef1_tingkat") if c in L.columns]]
L.to_csv(OUT_LATIH, index=False)
print(f"tersimpan: {OUT_LATIH}  baris {len(L):,} (" + ", ".join(f"{s} {n:,}" for s, n in L.sumber.value_counts().items())
      + f"); {time.time() - T0:.0f} dtk")
print("selesai.")
