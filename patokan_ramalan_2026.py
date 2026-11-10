"""
Ramalan relasi tingkat 2026 per kabupaten, R48(f')(iii) (tambahan bertanggal keempat Hasil_Patokan_21.md).

Membaca L dan e_k per tahun dari log R1 (fase R, Hasil_Patokan_24) yang sudah ada, sasaran per kabupaten:
    DL_FIRE_NASIONAL/penuh/patokan_tingkat_r1_kab.log   (VIIRS 2020-2025, ambang tahun latih)
    DL_FIRE_MODIS/patokan_modis_r_kab.log               (MODIS maju 2008-2019)
Regresi L atas e pada tahun dengan e > 0, intersep per rancangan; selang prakiraan 95 % OLS (df n - 3) untuk tahun
VIIRS baru. Tahun dengan e <= 0 dilaporkan sebagai dasar batas |L| <= 0,05.

Pakai:
    python patokan_ramalan_2026.py > DL_FIRE_NASIONAL/penuh/patokan_ramalan_2026.log
"""
import re

import numpy as np
from scipy import stats

POLA = re.compile(r"^\s+(\d{4}): L ([+-][\d.]+)\s+e_k ([+-][\d.]+)")


def baca(f, rancangan):
    out = []
    for ln in open(f, encoding="utf-8"):
        m = POLA.match(ln)
        if m:
            out.append((rancangan, int(m[1]), float(m[2]), float(m[3])))
    return out


T = baca("DL_FIRE_NASIONAL/penuh/patokan_tingkat_r1_kab.log", "VIIRS") + baca("DL_FIRE_MODIS/patokan_modis_r_kab.log", "MODIS")
assert len(T) == 18, len(T)
rendah = [t for t in T if t[3] <= 0]
print("Tahun dengan e <= 0 (kalender terlalu rendah):")
for r, th, L, e in rendah:
    print(f"  {r} {th}: L {L:+.3f}  e {e:+.3f}")
print(f"  rentang L [{min(t[2] for t in rendah):+.3f}, {max(t[2] for t in rendah):+.3f}]; maks |L| {max(abs(t[2]) for t in rendah):.3f}")

tinggi = [t for t in T if t[3] > 0]
X = np.array([[t[3], t[0] == "VIIRS", t[0] == "MODIS"] for t in tinggi], float)
y = np.array([t[2] for t in tinggi])
n, k = X.shape
b = np.linalg.lstsq(X, y, rcond=None)[0]
res = y - X @ b
s2 = (res ** 2).sum() / (n - k)
XtXi = np.linalg.inv(X.T @ X)
q = stats.t.ppf(0.975, n - k)
se_b = np.sqrt(s2 * XtXi[0, 0])
print(f"\nTahun dengan e > 0: n = {n} (VIIRS {int(X[:, 1].sum())}, MODIS {int(X[:, 2].sum())}), df = {n - k}")
print(f"  L = {b[1]:+.4f} (VIIRS) / {b[2]:+.4f} (MODIS) + {b[0]:.4f} e   kemiringan [{b[0] - q * se_b:+.4f}, {b[0] + q * se_b:+.4f}]")
print(f"  simpangan baku sisa {np.sqrt(s2):.4f}; t(0,975; {n - k}) = {q:.4f}")
print(f"  (X'X)^-1 untuk [e, VIIRS, MODIS]:\n{np.array2string(XtXi, precision=6)}")
print("\nSelang prakiraan 95 % untuk tahun VIIRS baru:")
for e in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5):
    x0 = np.array([e, 1.0, 0.0])
    yh = x0 @ b
    sp = np.sqrt(s2 * (1 + x0 @ XtXi @ x0))
    print(f"  e = {e:.2f}: L {yh:+.3f} [{yh - q * sp:+.3f}, {yh + q * sp:+.3f}]")
print("selesai.")
