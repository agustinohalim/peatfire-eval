"""
Selang bootstrap untuk uji mekanisme artikel Kalimantan (Hasil_Kalimantan_4 B), dari
DL_FIRE_NASIONAL/prediksi_kalimantan.csv (dibuat kalimantan_prediksi.py).

Pertama memeriksa bahwa berkas prediksi mereproduksi angka Hasil_Kalimantan_2/4 (GB ID per
provinsi), lalu menghitung selisih berpasangan dengan dua skema bootstrap 2.000 ulangan:
baris kabupaten-bulan (iid) dan klaster kabupaten.

  (ID+PROV) - ID      : apakah identitas provinsi menaikkan skor
  LOKAL - (ID+PROV)   : selisih pelatihan lokal yang tersisa

Pakai:  python kalimantan_selang.py DL_FIRE_NASIONAL/prediksi_kalimantan.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

P = pd.read_csv(sys.argv[1])
P = P[P["skema"] == "rolling"]
ACUAN = {("gabungan", "KalimantanBarat"): 0.749, ("gabungan", "KalimantanSelatan"): 0.572,
         ("gabungan", "KalimantanTengah"): 0.516, ("kabupaten", "KalimantanBarat"): 0.624,
         ("kabupaten", "KalimantanSelatan"): 0.772}


def selang(d, a, b, n=2000, seed=2026):
    """Selisih AUC-PR (a - b) dengan selang iid dan klaster kabupaten; baris NaN dibuang."""
    d = d[d[a].notna() & d[b].notna()]
    y, sa, sb = d["y"].to_numpy(), d[a].to_numpy(), d[b].to_numpy()
    kab = d["kabupaten"].to_numpy()
    uk = np.unique(kab); idx = {k: np.where(kab == k)[0] for k in uk}
    rng = np.random.default_rng(seed)
    iid, kl = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            iid.append(ap(y[i], sa[i]) - ap(y[i], sb[i]))
        j = np.concatenate([idx[k] for k in rng.choice(uk, size=len(uk), replace=True)])
        if y[j].sum():
            kl.append(ap(y[j], sa[j]) - ap(y[j], sb[j]))
    return ap(y, sa) - ap(y, sb), np.percentile(iid, [2.5, 97.5]), np.percentile(kl, [2.5, 97.5])


print("Pemeriksaan reproduksi (GB ID per provinsi):")
for (s, p), v in ACUAN.items():
    d = P[(P["sasaran"] == s) & (P["provinsi"] == p)]
    nilai = ap(d["y"], d["gradient-boosting"])
    print(f"  {s:<10}{p:<18} {nilai:.3f}  acuan {v:.3f}  {'OK' if abs(nilai - v) < 0.0015 else 'BEDA'}")

for s in ("gabungan", "kabupaten"):
    print(f"\nSASARAN {s.upper()}")
    print(f"  {'wilayah':<20}{'(ID+PROV)-ID':>13}  {'iid':<18}{'klaster':<18}{'LOKAL-(ID+PROV)':>16}  {'klaster':<18}")
    dS = P[P["sasaran"] == s]
    for p in ["Seluruh Kalimantan"] + sorted(dS["provinsi"].unique()):
        d = dS if p.startswith("Seluruh") else dS[dS["provinsi"] == p]
        if d["y"].sum() < 5:
            continue
        b1, i1, k1 = selang(d, "gb_prov", "gradient-boosting")
        teks2 = ""
        if not p.startswith("Seluruh"):
            b2, _, k2 = selang(d, "gb_lokal", "gb_prov")
            teks2 = f"{b2:>+16.3f}  [{k2[0]:+.3f}, {k2[1]:+.3f}]"
        print(f"  {p:<20}{b1:>+13.3f}  [{i1[0]:+.3f}, {i1[1]:+.3f}]  [{k1[0]:+.3f}, {k1[1]:+.3f}]{teks2}")
