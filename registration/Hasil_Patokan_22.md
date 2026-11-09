# Hasil Patokan 22 — fase P revisi (bagian tingkat dengan ambang dari tahun latih)

**9 November 2026.** Lanjutan `Hasil_Patokan_21.md`, untuk `Naskah_Artikel2_Revisi.md`.
Pemicu: telaah internal pra-kirim kedua puluh dua atas draf revisi (catatan telaah tidak dirilis),
butir metode dan advokat setan: bagian tingkat utama naskah (GB atas kalender mentah, VIIRS S1 2020–2025:
0,87 per kabupaten, 0,62 nasional; pecahan periode/tahun 0,60/0,27) memakai ambang dari seluruh panel, termasuk
tahun uji, dan belum pernah dihitung ulang dengan ambang tahun latih, padahal Tabel S6b (fase G2) dan protokol 2026
memakai ambang tahun latih. Ambang dari seluruh panel menyamakan prevalensi latih dan uji lebih dari ambang tahun
latih, jadi arah pengaruhnya tidak jelas sebelum dihitung.

## Rancangan

**Data dan prakiraan** persis fase G2 (`patokan_revisi_ambang_peluang.py`, Hasil_Patokan_12): label ulang per
lipatan S1 (persentil 90 kabupaten dari tahun latih, batas bawah 10; persentil 90 gabungan dari tahun latih); RF
dan GB dari `patokan_prediksi_kepekaan_penuh.csv` (dilatih ulang pada label ulang, benih 42); kalender EF-1m / EF-3m
dari `R.klim_frek` pada label ulang tahun latih; EF-L fase E/F pada label ulang; per kabupaten hanya kabupaten
dengan positif di tahun latih lipatan. Pemeriksaan: BSS GB atas kalender harus sama dengan G2 (+0,393 per
kabupaten, +0,273 nasional).

**Pemecahan** persis fase I4 (`patokan_tingkat_i.py`): (i) mentah; (ii) offset periode — satu geser logit untuk
semua baris uji semua tahun sehingga rata-rata prakiraan = laju teramati gabungan, untuk model dan rujukan
masing-masing; (iii) geser per tahun ke laju teramati tahun itu. Bagian periode = 1 − BSS(ii)/BSS(i); bagian
tahun = (BSS(ii) − BSS(iii))/BSS(i); bagian tingkat total = 1 − BSS(iii)/BSS(i). Rata-rata rasio tahunan, dua tahap
(1.000 ulangan, benih 2026), untuk GB dan RF atas kalender mentah, kedua sasaran. Skrip baru
`patokan_tingkat_p.py` menyalin bagian data G2 dan fungsi I4 tanpa perubahan isi.

**Deskriptif tambahan (tanpa aturan).** Prevalensi uji tiap tahun dan prevalensi latih rata-rata per lipatan
dengan kedua ambang; MDD80 BSS GB dan RF atas EF-L dengan ambang tahun latih (enam tahun, df 5), untuk
dibandingkan dengan 0,11–0,24 di naskah (ambang seluruh panel).

## Aturan keputusan (R49)

Untuk GB atas kalender mentah (angka yang dikutip abstrak), tiap sasaran, dengan ambang tahun latih:

- **R49(a)** Abstrak, §1, §6.2 dan §9 memakai bagian tingkat total dengan ambang tahun latih sebagai angka utama,
  karena rancangan itu tidak membiarkan tahun uji menentukan ambang; angka ambang seluruh panel ditulis di
  sampingnya di §6.2 (dan Tabel/Gambar yang memuatnya tetap, dengan keterangan). Bila selisih keduanya < 0,05,
  abstrak boleh tetap memakai satu angka dengan kalimat "with thresholds from training years as well".
- **R49(b)** Per kabupaten: "mostly" tetap bila bagian total ≥ 0,5; "much of" bila 0,25 ≤ bagian < 0,5; bila
  < 0,25, klaim bahwa sebagian besar perolehan peluang per kabupaten adalah tingkat api gugur dan abstrak ditulis
  ulang. Nasional diperlakukan sama, tanpa kata "mostly" di abstrak kecuali bagian nasional ≥ 0,5.
- **R49(c)** Bila vonis R49(b) berubah untuk salah satu sasaran, panel 23 dijalankan; bila tidak, tidak.
- **R49(d)** Pecahan periode/tahun dilaporkan di §6.2 bersama angka lama; tidak ada klaim baru dari pecahan ini.
- RF dilaporkan di suplemen, tanpa aturan.

## Hasil (dijalankan 9 Nov 2026 sesudah komit 86ec90f dan 745b6fb)

Log: `DL_FIRE_NASIONAL/penuh/patokan_tingkat_p_{kab,nas}.log`. Pemeriksaan: BSS GB atas kalender +0,3929 per
kabupaten dan +0,2725 nasional = G2; RF +0,3835 / +0,2934; EF-L +0,3619 / +0,2555. Per kabupaten 29.376 baris
(kabupaten dengan positif di tahun latih), prevalensi 3,75 % / 5,39 %.

| Ambang tahun latih, atas kalender mentah | Per kabupaten | Nasional |
|---|---|---|
| GB mentah / sesudah offset periode / sesudah geser per tahun | +0,393 / +0,146 / +0,042 | +0,273 / +0,121 / +0,116 |
| GB bagian periode [dua tahap] | 0,63 [0,30, 0,95] | 0,56 [0,02, 0,96] |
| GB bagian tahun | 0,26 [0,06, 0,45] | 0,02 [−0,67, 0,38] |
| **GB bagian tingkat total** | **0,89 [0,72, 1,05]** | **0,57 [0,06, 0,77]** |
| RF bagian periode / tahun / total | 0,62 / 0,29 / 0,92 [0,81, 1,03] | 0,50 / 0,21 / 0,71 [0,39, 0,87] |
| Ambang seluruh panel (Hasil_Patokan_14, I4): GB total; periode / tahun | 0,87; 0,60 / 0,27 | 0,62; 0,55 / 0,07 |

Geser periode (logit), ambang tahun latih: GB −0,289, RF −0,532, kalender −1,140 per kabupaten; −0,329, −0,411,
−0,978 nasional. Prevalensi latih rata-rata per lipatan dengan ambang tahun latih 6,8–7,4 % (per kabupaten) dan
9,5–9,8 % (nasional), dengan ambang seluruh panel 7,1–9,7 % dan 9,8–12,4 %; prevalensi uji hampir sama untuk
kedua ambang (per kabupaten 0,89–9,30 % lawan 0,89–9,55 %). Dengan ambang tahun latih selisih tingkat latih–uji
sedikit lebih kecil di 2020–2022, dan bagian tingkat per kabupaten tidak turun.

MDD80 BSS atas EF-L, ambang tahun latih (enam tahun, df 5): GB 0,123 per kabupaten, 0,265 nasional; RF 0,109 / 0,090
(naskah, ambang seluruh panel: 0,11–0,24).

**R49.** (a) Selisih GB total: per kabupaten 0,02 (< 0,05), nasional 0,05 (tidak < 0,05) → angka utama abstrak, §1,
§6.2 dan §9 memakai ambang tahun latih: 89 % per kabupaten, 57 % nasional; angka ambang seluruh panel (87 %, 62 %)
ditulis di sampingnya di §6.2, Tabel/Gambar tetap dengan keterangan. (b) Per kabupaten 0,89 ≥ 0,5 → "mostly" tetap;
nasional 0,57 ≥ 0,5, abstrak tidak memakai "mostly" nasional (tidak berubah). (c) Vonis tidak berubah → tanpa panel 23.
(d) Pecahan periode/tahun dilaporkan di §6.2; per kabupaten hampir sama (0,63/0,26 lawan 0,60/0,27), nasional juga
(0,56/0,02 lawan 0,55/0,07); bagian tahun nasional tetap tak terbedakan dari nol.
