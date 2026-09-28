"""
Percobaan Kalimantan 2 — apakah model yang dipelajari bisa dipindah antarprovinsi?

Untuk tiap provinsi uji P dan tiap tahun uji Y (2019-2025, rolling-origin), tiga cara melatih
model yang dipelajari (regresi logistik ONI, regresi logistik penuh, gradient boosting):

  ID    : dilatih pada seluruh Kalimantan tahun < Y, termasuk sejarah P   (dalam-domain)
  LOPO  : dilatih pada empat provinsi lain tahun < Y, tanpa P sama sekali (transfer)
  LOKAL : dilatih pada sejarah P sendiri tahun < Y saja

Baseline yang tidak dipelajari (klimatologi, persistence, seasonal naive, ratio scaling) selalu
dihitung dari sejarah P sendiri: klimatologi butuh sejarah kabupaten itu, jadi "memindahkannya"
tidak bermakna. Dengan demikian yang diuji adalah apakah model yang belajar dari provinsi lain
masih mengungguli baseline lokal yang murah.

Dua sasaran: ambang gabungan Kalimantan (p90) dan ambang per kabupaten (p90).

Pakai:
    python percobaan_kalimantan_transfer.py DL_FIRE_NASIONAL/panel_kalimantan.csv DL_FIRE_SV-C2_792597/oni.ascii.txt

(panel_kalimantan.csv dibuat oleh percobaan_kalimantan.py.)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")      # peringatan konvergensi regresi logistik, lihat Hasil_Kalimantan_1

BASE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(BASE, "percobaan.py"), encoding="utf-8").read().split("if __name__")[0])

jalur_kal, oni_path = sys.argv[1], sys.argv[2]
df, ambang = muat(jalur_kal, oni_path)
TAHUN_UJI = list(range(2019, 2026))
DIPELAJARI = ["regresi-ONI", "regresi-penuh", "gradient-boosting"]
BASELINE = ["klimatologi", "persistence", "seasonal-naive", "rasio-analog"]
GB, KL = "gradient-boosting", "klimatologi"
LABEL = {"klimatologi": "Climatology", "gradient-boosting": "Gradient boosting",
         "persistence": "Persistence", "seasonal-naive": "Seasonal naive",
         "rasio-analog": "Ratio scaling", "regresi-ONI": "Logistic, ONI",
         "regresi-penuh": "Logistic, full"}
PROV = sorted(df["provinsi"].unique())


def jalankan(d):
    """Kembalikan dict (provinsi, cara) -> {'y': ..., model: skor} atas seluruh tahun uji."""
    hasil = {(p, c): {"y": []} for p in PROV for c in ("ID", "LOPO", "LOKAL")}
    for th in TAHUN_UJI:
        lalu = d[d["tahun"] < th]
        sk_id = None
        for p in PROV:
            uji = d[(d["tahun"] == th) & (d["provinsi"] == p)]
            if len(uji) == 0:
                continue
            s_id = skor_model(lalu, uji)                                   # ID, dan baseline lokal
            s_lopo = skor_model(lalu[lalu["provinsi"] != p], uji)
            lokal = lalu[lalu["provinsi"] == p]
            s_lok = skor_model(lokal, uji) if lokal["y"].nunique() == 2 else None
            y = uji["y"].to_numpy()
            for c, s in (("ID", s_id), ("LOPO", s_lopo), ("LOKAL", s_lok)):
                h = hasil[(p, c)]
                h["y"].append(y)
                for n in BASELINE:
                    h.setdefault(n, []).append(s_id[n][1])
                for n in DIPELAJARI:
                    sumber = s if s is not None else s_id
                    h.setdefault(n, []).append(sumber[n][1] if s is not None else np.full(len(y), np.nan))
    for k, h in hasil.items():
        for n in h:
            h[n] = np.concatenate(h[n])
    return hasil


def ap(y, s):
    ok = ~np.isnan(s)
    if ok.sum() == 0 or y[ok].sum() == 0:
        return np.nan
    return average_precision_score(y[ok], s[ok])


def boot(y, a, b, n=2000, seed=2026):
    rb = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rb.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        out.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    return np.percentile(out, [2.5, 97.5])


def peringkat(nilai):
    urut = sorted(nilai, key=lambda n: -nilai[n])
    return {n: urut.index(n) + 1 for n in nilai}


def laporan(hasil, judul):
    print("\n" + "=" * 96)
    print(judul)
    print("=" * 96)
    print(f"{'provinsi':<18}{'pos':>4}  {'GB ID':>6}{'GB LOPO':>8}{'GB LOKAL':>9}{'klim':>6}  "
          f"{'GB LOPO - klim [95%]':<24}{'rank GB (ID/LOPO/LOKAL)':>24}{'tau ID-LOPO':>12}")
    ringkas = []
    for p in PROV:
        y = hasil[(p, "ID")]["y"]
        if y.sum() < 5:
            print(f"{p:<18}{int(y.sum()):>4}  terlalu sedikit positif"); continue
        nilai = {c: {n: ap(y, hasil[(p, c)][n]) for n in BASELINE + DIPELAJARI} for c in ("ID", "LOPO", "LOKAL")}
        rk = {c: peringkat({n: v for n, v in nilai[c].items() if not np.isnan(v)}) for c in nilai}
        lo, hi = boot(y, hasil[(p, "LOPO")][GB], hasil[(p, "ID")][KL])
        rid, rlo = rk["ID"], rk["LOPO"]
        t = kendalltau([rid[n] for n in rid], [rlo[n] for n in rid]).statistic
        gb_lok = nilai["LOKAL"][GB]
        print(f"{p:<18}{int(y.sum()):>4}  {nilai['ID'][GB]:>6.3f}{nilai['LOPO'][GB]:>8.3f}"
              f"{gb_lok:>9.3f}{nilai['ID'][KL]:>6.3f}  "
              f"{nilai['LOPO'][GB] - nilai['ID'][KL]:+.3f} [{lo:+.3f}, {hi:+.3f}]    "
              f"{rid[GB]:>6}/{rlo[GB]}/{rk['LOKAL'].get(GB, '-')}{t:>14.3f}")
        ringkas.append((p, nilai["ID"][GB] - nilai["LOPO"][GB]))
    kerugian = [x for _, x in ringkas]
    print(f"\n  kerugian transfer GB (AUC-PR ID - LOPO): median {np.median(kerugian):+.3f}, "
          f"rentang {min(kerugian):+.3f} s/d {max(kerugian):+.3f}")


laporan(jalankan(df), f"A. AMBANG GABUNGAN KALIMANTAN (p90 > {ambang:.0f})")

mentah = pd.read_csv(jalur_kal)
q_kab = mentah[mentah["tahun"] <= 2025].groupby("kabupaten")["titik_panas"].quantile(0.90)
d2 = df.copy()
d2["y"] = (d2["titik_panas"] > d2["kabupaten"].map(q_kab)).astype(int)
laporan(jalankan(d2), "B. AMBANG PER KABUPATEN (p90)")
print("\nselesai.")
