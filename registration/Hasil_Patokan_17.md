# Hasil Tolok Ukur Nasional 17 — fase L revisi (agregasi tahun untuk hasil peluang VIIRS)

**8 November 2026.** Lanjutan `Hasil_Patokan_16.md` (fase K), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim ketiga belas atas draf revisi (catatan telaah tidak
dirilis). Temuan utamanya: klaim berarah per kabupaten dalam prakiraan maju MODIS ("beats EF-L in the other
nine") bertumpu pada subset tahun yang dipilih menurut prevalensi teramati, dan salah hitung (2009 −0,003).
Yang perlu analisis: hasil pokok VIIRS atas EF-L (Tabel 4) dan bagian tingkat 87 % belum diuji dengan satu rasio
atas semua baris, padahal 2021 dan 2022 sama sepinya (1–2 %) dengan tahun MODIS yang membalik vonis maju.

## Rancangan fase L (ditulis 8 Nov 2026, SEBELUM skrip fase L dijalankan)

Pasca-hoc; angka rata-rata rasio Tabel 4 dan per tahunnya sudah terlihat sejak fase E–I. Data, prediksi tersimpan
fase E, kolom model dan rujukan, geser logit dan pemecahan periode/tahun sama dengan `patokan_tingkat_i.py`.
Skrip baru `patokan_tingkat_l.py` (satu sasaran per jalan). Semua kabupaten, enam tahun biasa 2020–2025 (S1).
Dua tahap 1.000 ulangan, benih bootstrap B.SEED.

**L1** Untuk setiap sel Tabel 4 (model dan rujukan atas kalender, EF-P, EF-L): (a) rata-rata rasio tahunan
(harus mereproduksi fase I), dua tahap + t95; (b) satu rasio atas semua baris uji, dua tahap; (c) median rasio
tahunan dan banyak tahun di atas nol. Dua sasaran.

**L2** Bagian tingkat untuk GB dan RF atas kalender dan EF-P sebagai satu rasio: BSS mentah, sesudah offset
periode, sesudah geser per tahun (semuanya satu rasio), bagian periode, bagian tahun, total, dua tahap. Dua sasaran.

**R40 (aturan keputusan, ditulis sebelum dijalankan).**
(a) Bila untuk model mana pun atas EF-L atau atas kalender satu rasio berbeda tanda dari rata-rata rasio, atau
selang dua tahapnya memberi vonis berbeda dari selang dua tahap rata-rata rasio, sel itu dilaporkan di §6.2 dan
Suplemen, dan setiap kalimat abstrak/§1/§9 yang bergantung pada sel itu menyebut kedua agregasi. Bila tidak ada,
satu kalimat di §6.2: satu rasio tidak mengubah tanda atau vonis dua tahap untuk Tabel 4.
(b) Bila bagian tingkat total GB atas kalender per kabupaten sebagai satu rasio berbeda lebih dari 0,15 dari 0,87,
abstrak menyebut kedua angka; bila tidak, satu kalimat di §6.2.
(c) Hasil satu rasio hanya punya selang dua tahap; naskah tidak menyebutnya "lolos" atau "beats", melainkan
"selang dua tahap mengecualikan nol".

**Aturan teks (tanpa analisis), dari telaah internal ketiga belas.** Vonis maju MODIS per kabupaten ditulis menurut agregasi yang dapat
ditetapkan sebelum musim (rata-rata rasio, satu rasio, median): GB dan EF-L tak terbedakan. Hasil "tanpa tiga
tahun tersepi" dan belahan 2008–2015 / 2016–2019 dilaporkan hanya sebagai diagnostik deskriptif, dengan catatan
bahwa subset pertama ditentukan oleh prevalensi teramati. Ini menyimpang dari R39a cabang ketiga, yang disusun
dengan subset itu; penyimpangan ditulis di Lampiran A.

## Hasil

Log (gitignored): `DL_FIRE_NASIONAL/penuh/patokan_tingkat_l_{kab,nas}.log`, skrip d598266, jalan 8 Nov 09:27.
Rata-rata rasio (a) mereproduksi Tabel 4 / fase I di setiap sel (mis. GB atas kalender per kabupaten +0,4149, atas
EF-L +0,0666; bagian tingkat rata-rata rasio 0,87 dari log fase I). Prevalensi per kabupaten 2020–2025: 3,49 /
1,68 / 1,07 / 11,58 / 3,75 / 2,55 %; nasional 5,29 / 4,32 / 3,33 / 11,43 / 6,58 / 5,82 %. Empat desimal dari program.

### L1 Sel yang disebut di naskah: (a) rata-rata rasio [dua tahap] t [..]; (b) satu rasio [dua tahap]; (c) median, tahun > 0

| Sel | Per kabupaten (a) | (b) | (c) | Nasional (a) | (b) | (c) |
|---|---|---|---|---|---|---|
| GB atas kalender | +0,4149 [+0,2301, +0,6051] t [+0,1469, +0,6830] | +0,4009 [+0,2205, +0,6003] | +0,3905 6/6 | +0,2554 [+0,1207, +0,4234] t [+0,0376, +0,4733] | +0,2526 [+0,1165, +0,4262] | +0,1738 6/6 |
| RF atas kalender | +0,4013 [+0,2253, +0,5875] t [+0,1411, +0,6616] | +0,3881 [+0,2110, +0,5779] | +0,4218 6/6 | +0,2711 [+0,1478, +0,4239] t [+0,0770, +0,4651] | +0,2717 [+0,1402, +0,4238] | +0,2579 6/6 |
| EF-L atas kalender | +0,3757 [+0,1696, +0,5739] t [+0,0829, +0,6686] | +0,3534 [+0,1292, +0,5745] | +0,3894 5/6 | +0,2318 [+0,1125, +0,3627] t [+0,0613, +0,4024] | +0,2363 [+0,1143, +0,3649] | +0,2141 6/6 |
| GB atas EF-P | +0,3830 [+0,2156, +0,5607] t [+0,1474, +0,6187] | +0,3570 [+0,2155, +0,5421] | +0,3277 6/6 | +0,2586 [+0,1243, +0,4284] t [+0,0394, +0,4779] | +0,2557 [+0,1191, +0,4288] | +0,1878 6/6 |
| GB atas EF-L | +0,0666 [−0,0199, +0,1524] t [−0,0475, +0,1807] | +0,0735 [−0,0310, +0,1571] | +0,0652 4/6 | +0,0372 [−0,0941, +0,1522] t [−0,1396, +0,2141] | +0,0213 [−0,1069, +0,1422] | +0,0722 5/6 |
| GB disetel atas EF-L | +0,0646 [−0,0511, +0,1749] t [−0,0840, +0,2131] | +0,0600 [−0,0709, +0,1604] | +0,1282 4/6 | +0,0690 [−0,0175, +0,1604] t [−0,0528, +0,1908] | +0,0540 [−0,0237, +0,1526] | +0,0817 5/6 |
| RF atas EF-L | +0,0304 [−0,0552, +0,1020] t [−0,0686, +0,1294] | +0,0537 [−0,0500, +0,1200] | +0,0519 4/6 | +0,0577 [−0,0002, +0,1266] t [−0,0225, +0,1379] | +0,0463 [−0,0052, +0,1192] | +0,0461 4/6 |
| Riwayat lokal EF-1m atas EF-L | −0,1465 [−0,3032, −0,0052] t [−0,3347, +0,0416] | −0,0602 [−0,2535, +0,0671] | −0,1575 1/6 | +0,0120 [−0,0653, +0,0806] t [−0,0761, +0,1001] | +0,0262 [−0,0527, +0,0865] | +0,0086 3/6 |
| MLP atas EF-L | −0,1258 [−0,3455, +0,0672] t [−0,4146, +0,1630] | −0,1122 [−0,3551, +0,0420] | −0,0184 2/6 | −0,0513 [−0,3737, +0,1498] t [−0,4498, +0,3473] | −0,0686 [−0,3891, +0,1425] | +0,0787 5/6 |
| GB + hitungan nasional atas EF-L | +0,0909 [+0,0071, +0,1798] t [−0,0217, +0,2034] | +0,1002 [+0,0002, +0,1780] | +0,0851 5/6 | +0,0654 [−0,0155, +0,1539] t [−0,0508, +0,1817] | +0,0511 [−0,0282, +0,1469] | +0,0904 5/6 |

Sel lain (semua model atas kalender dan EF-P, konstan, EF-P atas kalender) di log; tak satu pun berubah tanda.

### L2 Bagian tingkat sebagai satu rasio [dua tahap]

| | Mentah | Sesudah offset periode | Sesudah geser per tahun | Bagian periode | Bagian tahun | Total |
|---|---|---|---|---|---|---|
| Per kab, GB atas kalender | +0,4009 | +0,1811 | +0,0682 | 0,55 [0,07, 0,91] | 0,28 [0,09, 0,52] | **0,83** [0,54, 1,03] |
| Per kab, RF atas kalender | +0,3881 | +0,1567 | +0,0560 | 0,60 [0,26, 0,85] | 0,26 [0,13, 0,42] | 0,86 [0,64, 1,01] |
| Per kab, GB atas EF-P | +0,3570 | +0,1840 | +0,0890 | 0,48 [−0,03, 0,94] | 0,27 [0,01, 0,47] | 0,75 [0,40, 1,02] |
| Per kab, RF atas EF-P | +0,3432 | +0,1597 | +0,0770 | 0,53 [0,15, 0,86] | 0,24 [0,09, 0,38] | 0,78 [0,49, 1,00] |
| Nasional, GB atas kalender | +0,2526 | +0,1270 | +0,0957 | 0,50 [−0,22, 0,96] | 0,12 [−0,46, 0,54] | 0,62 [0,19, 0,79] |
| Nasional, RF atas kalender | +0,2717 | +0,1365 | +0,0806 | 0,50 [−0,03, 0,71] | 0,21 [0,02, 0,48] | 0,70 [0,37, 0,85] |

(Rata-rata rasio fase I: per kab GB 0,87 = 0,60 + 0,27; nasional GB 0,62 = 0,55 + 0,07.)

### Vonis R40

**R40a**: tak ada model atas kalender atau EF-L yang berubah tanda. Satu sel berubah vonis dua tahap: riwayat lokal
EF-1m atas EF-L per kabupaten, rata-rata rasio −0,147 (dua tahap mengecualikan nol, †) tetapi satu rasio −0,060
[−0,254, +0,067] (memuat nol). Dilaporkan di §6.2 dan Suplemen; kalimat naskah yang memakai sel ini (§6.2 "below
EF-L under the two-stage interval") menyebut kedua agregasi. Tak ada kalimat abstrak/§1/§9 yang bergantung padanya.
GB + hitungan nasional atas EF-L per kabupaten: dua tahap mengecualikan nol di kedua agregasi (+0,091 / +0,100),
tingkat tahun tidak — vonis sama.
**R40b**: bagian tingkat total GB atas kalender per kabupaten sebagai satu rasio 0,83 (selisih 0,04 dari 0,87,
< 0,15) → satu kalimat di §6.2. Pembagian periode/tahun bergeser sedikit (0,55 / 0,28 lawan 0,60 / 0,27).
**R40c**: diterapkan di teks.

### Bacaan

Berbeda dengan prakiraan maju MODIS, hasil peluang VIIRS tidak bergantung pada cara rata-rata: model pembelajaran
di atas kalender dan EF-P, tak terbedakan dari EF-L, dan bagian tingkat 0,83–0,87, di ketiga agregasi. Tahun sepi
2021 dan 2022 tidak membalik apa pun di sini.
