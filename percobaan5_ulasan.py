"""
Percobaan 5 — menjawab tiga keberatan tinjauan mandiri atas naskah EMS.

  A. Kalibrasi. Naskah memasang regresi isotonik pada skor data LATIH. Untuk XGBoost skor
     latih itu terlalu pas (overfit), jadi pemetaan isotonik bisa mendorong prediksi uji ke
     ujung. Di sini kalibrator dipasang pada prediksi out-of-fold: untuk tahun uji Y, tiga
     tahun sebelumnya (Y-3..Y-1) masing-masing diprediksi oleh model yang dilatih pada
     tahun-tahun sebelum tahun itu, lalu isotonik dipasang pada prediksi tersebut.

  B. Ketidakpastian protokol operasional. Naskah menyebut AUC-PR "deterministik". Di sini
     selang bootstrap 2000 ulangan untuk AUC-PR tiap model dan untuk selisih gradient
     boosting - klimatologi, dengan dua skema: baris district-month (iid) dan klaster
     kabupaten (14 klaster).

  C. Klaim yang mungkin dalam derau.
     C1. Per tahun: selang bootstrap selisih skill tiap tahun, uji tanda, dan "faktor
         sebelas" dihitung ulang dalam satuan yang sama (skill ternormalisasi).
     C2. Ambang 80 dan 95: berapa persen undian penyeimbangan yang menaruh gradient
         boosting di atas klimatologi menurut ROC-AUC.

Tidak menyunting naskah. Keluaran dibaca manusia dulu; putusan klaim ada di tangan penulis.

Pakai:
    python percobaan5_ulasan.py DL_FIRE_SV-C2_792597/panel_bulanan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt
"""

import os
import sys
import numpy as np
import pandas as pd

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "percobaan.py"),
          encoding="utf-8").read().split("if __name__")[0])

panel_path, oni_path = sys.argv[1], sys.argv[2]
df, ambang = muat(panel_path, oni_path)
TAHUN_UJI = list(range(2019, 2026))
GB, KL = "gradient-boosting", "klimatologi"
B_BOOT = 2000


def skill(y, s):
    prev = y.mean()
    return (average_precision_score(y, s) - prev) / (1 - prev)


# ============================================================ A. kalibrasi out-of-fold

print("=" * 78)
print("A. KALIBRASI — isotonik pada skor latih (naskah) lawan pada prediksi out-of-fold")
print("=" * 78)

nama = None
kum = {}
for th in TAHUN_UJI:
    latih, uji = df[df["tahun"] < th], df[df["tahun"] == th]
    skor = skor_model(latih, uji)
    if nama is None:
        nama = list(skor.keys())
        kum = {n: {"y": [], "raw": [], "cal_latih": [], "cal_oof": []} for n in nama}

    # prediksi out-of-fold untuk tiga tahun sebelum th
    oof_s = {n: [] for n in nama}
    oof_y = []
    for dalam in range(th - 3, th):
        l2, u2 = df[df["tahun"] < dalam], df[df["tahun"] == dalam]
        s2 = skor_model(l2, u2)
        for n in nama:
            oof_s[n].append(s2[n][1])
        oof_y.append(u2["y"].to_numpy())
    oof_y = np.concatenate(oof_y)

    y = uji["y"].to_numpy()
    for n, (sl, su) in skor.items():
        iso_l = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(sl, latih["y"].to_numpy())
        iso_o = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.concatenate(oof_s[n]), oof_y)
        kum[n]["y"].append(y)
        kum[n]["raw"].append(su)
        kum[n]["cal_latih"].append(iso_l.predict(su))
        kum[n]["cal_oof"].append(iso_o.predict(su))

for n in nama:
    for k in kum[n]:
        kum[n][k] = np.concatenate(kum[n][k])
Y = kum[nama[0]]["y"]

print(f"n uji = {len(Y)}, positif = {int(Y.sum())}, prevalensi = {Y.mean() * 100:.1f}%\n")
print(f"{'model':<20}{'AUC-PR':>8}   {'Brier latih':>11}{'ECE latih':>10}   {'Brier OOF':>10}{'ECE OOF':>9}")
print("-" * 78)
tabA = []
for n in nama:
    r = kum[n]
    baris = (n, average_precision_score(Y, r["raw"]),
             brier_score_loss(Y, r["cal_latih"]), ece(Y, r["cal_latih"]),
             brier_score_loss(Y, r["cal_oof"]), ece(Y, r["cal_oof"]))
    tabA.append(baris)
    print(f"{baris[0]:<20}{baris[1]:>8.3f}   {baris[2]:>11.4f}{baris[3]:>10.4f}   {baris[4]:>10.4f}{baris[5]:>9.4f}")

for kol, label in ((3, "latih"), (5, "OOF")):
    urut = sorted(tabA, key=lambda b: b[kol])
    print(f"  ECE {label:<5}: terbaik {urut[0][0]} ({urut[0][kol]:.4f}), "
          f"terburuk {urut[-1][0]} ({urut[-1][kol]:.4f}), peringkat GB "
          f"{[b[0] for b in urut].index(GB) + 1} dari {len(urut)}")

p_gb = kum[GB]["cal_latih"]
print(f"\n  sebaran prediksi terkalibrasi GB (latih): {len(np.unique(np.round(p_gb, 4)))} nilai unik; "
      f"persentil 50/90/99 = {np.percentile(p_gb, 50):.3f}/{np.percentile(p_gb, 90):.3f}/{np.percentile(p_gb, 99):.3f}")
p_gb = kum[GB]["cal_oof"]
print(f"  sebaran prediksi terkalibrasi GB (OOF)  : {len(np.unique(np.round(p_gb, 4)))} nilai unik; "
      f"persentil 50/90/99 = {np.percentile(p_gb, 50):.3f}/{np.percentile(p_gb, 90):.3f}/{np.percentile(p_gb, 99):.3f}")

# ============================================================ B. bootstrap operasional

print("\n" + "=" * 78)
print(f"B. SELANG BOOTSTRAP AUC-PR PREVALENSI ASLI ({B_BOOT} ulangan)")
print("=" * 78)

rng = np.random.default_rng(2026)
kab = np.concatenate([df[df["tahun"] == th]["kabupaten"].to_numpy() for th in TAHUN_UJI])
kab_unik = np.unique(kab)
idx_kab = {k: np.where(kab == k)[0] for k in kab_unik}


def boot(skema):
    out = {n: [] for n in nama}
    beda = []
    for _ in range(B_BOOT):
        if skema == "iid":
            i = rng.integers(0, len(Y), len(Y))
        else:
            pilih = rng.choice(kab_unik, size=len(kab_unik), replace=True)
            i = np.concatenate([idx_kab[k] for k in pilih])
        if Y[i].sum() == 0:
            continue
        v = {n: average_precision_score(Y[i], kum[n]["raw"][i]) for n in nama}
        for n in nama:
            out[n].append(v[n])
        beda.append(v[GB] - v[KL])
    return {n: np.array(x) for n, x in out.items()}, np.array(beda)


for skema in ("iid", "kabupaten"):
    out, beda = boot(skema)
    print(f"\n  skema {skema}:")
    for n in sorted(nama, key=lambda n: -average_precision_score(Y, kum[n]["raw"])):
        lo, hi = np.percentile(out[n], [2.5, 97.5])
        print(f"    {n:<20} {average_precision_score(Y, kum[n]['raw']):.3f}  [{lo:.3f}, {hi:.3f}]")
    lo, hi = np.percentile(beda, [2.5, 97.5])
    print(f"    selisih GB - klimatologi: {np.mean(beda):+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
          f"P(selisih <= 0) = {(beda <= 0).mean():.3f}")

# ============================================================ C1. per tahun

print("\n" + "=" * 78)
print("C1. PER TAHUN — selisih skill ternormalisasi GB - klimatologi")
print("=" * 78)

per = []
for th in TAHUN_UJI:
    m = np.concatenate([np.full((df["tahun"] == t).sum(), t) for t in TAHUN_UJI]) == th
    per.append((str(th), Y[m], kum[GB]["raw"][m], kum[KL]["raw"][m]))
for th in (2014, 2015):
    sisa, uji = df[df["tahun"] != th], df[df["tahun"] == th]
    s = skor_model(sisa, uji)
    per.append((f"{th}*", uji["y"].to_numpy(), s[GB][1], s[KL][1]))

print(f"{'tahun':<7}{'pos':>5}{'GB':>8}{'klim':>8}{'selisih':>9}   {'95% bootstrap':>18}  terbedakan?")
print("-" * 78)
tanda = []
beda_th = []
for lbl, y, sg, sk in per:
    d = skill(y, sg) - skill(y, sk)
    beda_th.append(d)
    bb = []
    for _ in range(B_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue
        bb.append(skill(y[i], sg[i]) - skill(y[i], sk[i]))
    lo, hi = np.percentile(bb, [2.5, 97.5])
    beda_nyata = "ya" if lo > 0 or hi < 0 else "tidak"
    tanda.append(np.sign(d) if beda_nyata == "ya" else 0)
    print(f"{lbl:<7}{int(y.sum()):>5}{skill(y, sg):>8.3f}{skill(y, sk):>8.3f}{d:>+9.3f}   "
          f"[{lo:+.3f}, {hi:+.3f}]  {beda_nyata}")

from math import comb
menang = sum(1 for d in beda_th if d > 0)
kalah = sum(1 for d in beda_th if d < 0)
k, nn = min(menang, kalah), menang + kalah
p_tanda = min(1.0, 2 * sum(comb(nn, j) for j in range(k + 1)) / 2 ** nn)
print(f"\n  GB unggul {menang}, klimatologi unggul {kalah} (tanpa ambang seri); "
      f"uji tanda dua sisi p = {p_tanda:.3f}")
print(f"  tahun dengan selang 95% yang tidak memuat nol: "
      f"GB {sum(1 for t in tanda if t > 0)}, klimatologi {sum(1 for t in tanda if t < 0)}")

agg_skill = skill(Y, kum[GB]["raw"]) - skill(Y, kum[KL]["raw"])
agg_ap = average_precision_score(Y, kum[GB]["raw"]) - average_precision_score(Y, kum[KL]["raw"])
maks = max(abs(d) for d in beda_th)
print(f"  selisih agregat: AUC-PR mentah {agg_ap:+.3f}, skill ternormalisasi {agg_skill:+.3f}")
print(f"  selisih per tahun terbesar (skill) {maks:.3f}; rasio dalam satuan sama = {maks / abs(agg_skill):.1f}; "
      f"rerata |selisih| per tahun {np.mean(np.abs(beda_th)):.3f} = {np.mean(np.abs(beda_th)) / abs(agg_skill):.1f}x")

# ============================================================ C2. ambang 80 / 95

print("\n" + "=" * 78)
print("C2. AMBANG — seberapa sering GB di atas klimatologi menurut ROC-AUC (300 undian)")
print("=" * 78)

mentah = pd.read_csv(panel_path)
mentah = mentah[mentah["tahun"] <= 2025]
assert abs(mentah["titik_panas"].quantile(0.90) - ambang) < 1e-9, "ambang tak cocok dengan muat()"

for q in (0.80, 0.90, 0.95):
    a = mentah["titik_panas"].quantile(q)
    d2 = df.copy()
    d2["y"] = (d2["titik_panas"] > a).astype(int)
    ys, sg, sk = [], [], []
    for th in TAHUN_UJI:
        s = skor_model(d2[d2["tahun"] < th], d2[d2["tahun"] == th])
        ys.append(d2[d2["tahun"] == th]["y"].to_numpy()); sg.append(s[GB][1]); sk.append(s[KL][1])
    ys, sg, sk = np.concatenate(ys), np.concatenate(sg), np.concatenate(sk)
    pos, neg = np.where(ys == 1)[0], np.where(ys == 0)[0]
    r2 = np.random.default_rng(7)
    atas, beda = 0, []
    for _ in range(300):
        i = np.concatenate([pos, r2.choice(neg, size=len(pos), replace=False)])
        rg, rk = roc_auc_score(ys[i], sg[i]), roc_auc_score(ys[i], sk[i])
        atas += rg > rk
        beda.append(rg - rk)
    lo, hi = np.percentile(beda, [2.5, 97.5])
    print(f"  persentil {int(q * 100)} (> {a:.0f}): GB di atas klimatologi pada {atas}/300 undian; "
          f"selisih ROC-AUC {np.mean(beda):+.4f} [{lo:+.4f}, {hi:+.4f}]")

print("\nselesai.")
