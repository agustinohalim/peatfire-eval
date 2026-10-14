"""
Uji tanda dan uji balik-tanda eksak atas tujuh selisih skill ternormalisasi per tahun
(GB - klimatologi), dibaca dari percobaan5_ulasan.log bagian C1. Untuk naskah JEPIN versi 4.

Pakai:
    python uji_tanda_jepin.py
"""
import itertools
import os
import re
from math import comb

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = next(d for d in (os.path.join(BASE, "DL_FIRE_NASIONAL", "oni_akhir"), os.path.join(BASE, "logs", "jepin"))
              if os.path.exists(os.path.join(d, "percobaan5_ulasan.log")))
log = open(os.path.join(LOGDIR, "percobaan5_ulasan.log"), encoding="cp1252", errors="replace").read()
d = np.array([float(x) for x in re.findall(r"^20(?:19|2[0-5])\s+\d+\s+\S+\s+\S+\s+([+-]\d\.\d+)", log, re.M)])
assert len(d) == 7, d
pos = int((d > 0).sum())
p_tanda = sum(comb(7, k) for k in range(8) if min(k, 7 - k) <= min(pos, 7 - pos)) / 128
obs = abs(d.mean())
p_balik = np.mean([abs((d * np.array(s)).mean()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=7)])
print(f"selisih per tahun: {d}")
print(f"positif {pos} dari 7; uji tanda dua sisi p = {p_tanda:.3f}")
print(f"rerata {d.mean():+.4f}; uji balik-tanda dua sisi p = {p_balik:.3f}")
