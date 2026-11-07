"""
PeatFireBench revisi fase A — cek PASCA-HOC (tidak didaftarkan dalam rancangan Hasil_Patokan_6).

Sesudah analisis terdaftar dijalankan, klim_frek1 (frekuensi bulan kalender itu saja, tanpa jendela)
ternyata lebih kuat daripada klim_frek terdaftar untuk sasaran per kabupaten (AUC-PR dalam-tahun 0,245
lawan 0,193). Cek:
  1. keunggulan rf, xgb, mlp atas klim_frek1 dalam-tahun S1 2020-2025, kedua sasaran [95 %, 1.000];
  2. S2 per tahun atas klim_frek1;
  3. kontras Temuan 1 (ekstrem S2 - tenang S1) atas klim_frek1, m1 dan m4, xgb, selang kabupaten 300.
Pakai: python patokan_revisi_pascahoc.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_pascahoc.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL_FIRE_NASIONAL")
P = pd.read_csv(os.path.join(D, "patokan_revisi_prediksi.csv"), dtype={"tag": str, "pulau": str})
REF = "klim_frek1"
SEED = 2026
BIASA = [str(t) for t in range(2020, 2026)]
EKSTREM = [("S2", "2015"), ("S2", "2014"), ("S2", "2019")]
TENANG = [("S1", t) for t in ("2020", "2021", "2022", "2024", "2025")]


def fmt(x, lo, hi):
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def dalam_tahun(d, a, b):
    return float(np.mean([ap(g.y, g[a]) - ap(g.y, g[b]) for _, g in d.groupby("tag") if g.y.sum()]))


def boot(d, a, b, n=1000):
    uk = d.gid.unique(); rng = np.random.default_rng(SEED)
    per = {t: g for t, g in d.groupby("tag")}
    idx = {t: {k: np.where(g.gid.to_numpy() == k)[0] for k in uk} for t, g in per.items()}
    arr = {t: (g.y.to_numpy(), g[a].to_numpy(float), g[b].to_numpy(float)) for t, g in per.items()}
    out = []
    for _ in range(n):
        pilih = rng.choice(uk, len(uk)); v = []
        for t in per:
            j = np.concatenate([idx[t][k] for k in pilih]); y, sa, sb = (x[j] for x in arr[t])
            if y.sum():
                v.append(ap(y, sa) - ap(y, sb))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


print("PASCA-HOC: rujukan klim_frek1 (bulan kalender itu saja)")
for sas in ("y_kabupaten", "y_gabungan"):
    d0 = P[P.sasaran == sas]
    if sas == "y_kabupaten":
        d0 = d0[d0.ada_positif == 1]
    b = d0[(d0.skema == "S1") & d0.tag.isin(BIASA)]
    print(f"\n--- {sas}, S1 2020-2025, dalam-tahun, unggul atas {REF} [95%, 1000]")
    for m in ("rf", "xgb", "mlp", "klim_frek", "log_klim_jeda"):
        print(f"  {m:<14} {fmt(dalam_tahun(b, m, REF), *boot(b, m, REF))}")
    print("  S2 per tahun, xgb atas klim_frek1 [95%, 500]:")
    for _, t in EKSTREM:
        dd = d0[(d0.skema == "S2") & (d0.tag == t)]
        print(f"    {t}: {fmt(dalam_tahun(dd, 'xgb', REF), *boot(dd, 'xgb', REF, 500))}")

print("\n--- Temuan 1 atas klim_frek1, xgb, 411 kabupaten, selang kabupaten 300")
Pk = P[P.ada_positif == 1]
blok = {(s,) + k: Pk[(Pk.sasaran == s) & (Pk.skema == k[0]) & (Pk.tag == k[1])]
        for s in ("y_kabupaten", "y_gabungan") for k in EKSTREM + TENANG}
kab = Pk.gid.unique()
idx = {k: {g: np.where(b.gid.to_numpy() == g)[0] for g in kab} for k, b in blok.items()}


def kontras(pilih):
    out = []
    for s in ("y_kabupaten", "y_gabungan"):
        r = []
        for grup in (EKSTREM, TENANG):
            v = []
            for k in grup:
                b = blok[(s,) + k]
                j = np.concatenate([idx[(s,) + k][g] for g in pilih]) if pilih is not None else np.arange(len(b))
                y, sx, sr = b.y.to_numpy()[j], b.xgb.to_numpy()[j], b[REF].to_numpy()[j]
                v.append([ap(y, sx) - ap(y, sr), auc(y, sx) - auc(y, sr)])
            r.append(np.mean(v, axis=0))
        out.append(r[0] - r[1])
    return np.array(out)  # [sasaran, ukuran]


titik = kontras(None)
rng = np.random.default_rng(SEED)
bs = np.array([kontras(rng.choice(kab, len(kab))) for _ in range(300)])
for u, nama in enumerate(("m1 AUC-PR mentah", "m4 ROC-AUC")):
    sel = bs[:, 0, u] - bs[:, 1, u]
    print(f"  {nama:<18} per kab {fmt(titik[0, u], *np.percentile(bs[:, 0, u], [2.5, 97.5]))}  "
          f"nasional {fmt(titik[1, u], *np.percentile(bs[:, 1, u], [2.5, 97.5]))}  "
          f"selisih {fmt(titik[0, u] - titik[1, u], *np.percentile(sel, [2.5, 97.5]))}")
print("\nselesai.")
