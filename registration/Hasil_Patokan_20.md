# Hasil Tolok Ukur Nasional 20 — fase O revisi (gradient boosting diberi masukan yang sama dengan EF-LD)

**9 November 2026.** Lanjutan `Hasil_Patokan_19.md` (fase N), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim keenam belas atas draf revisi (catatan telaah tidak
dirilis),
butir konsensus (metode, advokat setan): model dan rujukan tidak mendapat masukan yang sama. EF-LD melihat
anomali Drought Code kabupaten bulan lalu; gradient boosting yang dibandingkan dengannya tidak (VIIRS: fitur
penuh dengan FWI tetapi tanpa DC; MODIS maju: riwayat api dan anomali nasional saja). Kemenangan nasional maju
ditarik karena GB tak terbedakan dari EF-LD (+0,026), sehingga perbandingan itu menguji masukan kekeringan
sekaligus pembelajarannya. Telaah: pasang ulang GB dengan anomali DC yang sama dan nilai atas EF-LD.

## Rancangan fase O (ditulis 9 Nov 2026, SEBELUM skrip fase O dijalankan)

Semua pasca-hoc: tahun uji VIIRS 2020–2025 sudah dinilai oleh setiap putaran, tahun maju MODIS 2008–2019 oleh
fase J, K, M dan N. Interval dan "lolos" sama dengan fase M/N: dua tahap 1.000 ulangan benih B.SEED (versi cepat
matriks galat) dan t atas tahun; lolos = batas bawah dua tahap dan batas bawah t keduanya > 0 (rata-rata rasio
tahunan). Tiap sel tiga agregasi: (a) rata-rata rasio tahunan [dua tahap] t95; (b) satu rasio [dua tahap];
(c) median tahun dan banyak tahun > 0.

**O1 VIIRS, tahun biasa 2020–2025 (S1), kedua sasaran.** Skrip baru `patokan_tingkat_o.py`. `xgb_nasdc` = GB
baku seperti `xgb_nas` fase F (`patokan_revisi_tingkat.py`: fitur `R.FITUR`, 300 pohon, depth 3, lr 0,05,
subsample dan colsample 0,9, reg_lambda 1, benih 42, label `SAS`, NaN → 0) ditambah dua kolom: `anomali_nasional`
(definisi fase F) dan `anomali_dc` (definisi fase M: log1p(DC bulan lalu) kabupaten − rata-rata kabupaten itu
pada bulan kalender sama di tahun latih; baris latih tanpa tahunnya sendiri; DC dari
`gfwed_komponen_MERRA2.CORRECTED.csv`, geser satu bulan, hilang diisi median kabupaten). Jadi `xgb_nasdc`
melihat semua yang dilihat EF-LD. Latih = tahun sebelum tahun uji. EF-LD dihitung ulang seperti fase M dan harus
sama dengan fase M (selisih maks < 1e-4 terhadap `efld` ulang di skrip yang sama; pemeriksaan reproduksi: BSS
xgb atas EF-LD harus sama dengan log fase M, empat desimal). Sel: `xgb_nasdc` atas EF-LD, EF-L, kalender pasangan,
EF-P; `xgb_nasdc` atas `xgb_nas` dan atas `xgb` (deskriptif: yang ditambahkan DC dan hitungan nasional di dalam
GB). Untuk `xgb_nasdc` atas EF-LD dan atas EF-L: batas bawah dua tahap lima benih {B.SEED, 1, 2, 3, 4} × 1.000 dan
10.000 ulangan. Keluaran prediksi `DL_FIRE_NASIONAL/patokan_tingkat_o_gb_<sasaran>.csv` (tidak masuk repo); log
`DL_FIRE_NASIONAL/penuh/patokan_tingkat_o_{kab,nas}.log`.

**O2 MODIS, prakiraan maju 2008–2019, kedua sasaran.** Skrip baru `patokan_modis_o.py` dari `patokan_modis_n.py`
(lipatan R sama persis: latih = tahun MODIS sebelum tahun uji, ambang dan kabupaten dari tahun latih, lantai 2).
`gb_dc` = GB fase N (fitur `X_gb`: jeda 1, 2, 3, 12 bulan, sin/cos bulan, ef1, label bulan lalu, anomali nasional;
enam benih {42, 1, 2, 3, 4, 5}, Brier dirata-rata atas benih) ditambah kolom anomali DC fase N (baris latih tanpa
tahunnya sendiri). GB tanpa DC dan semua rujukan dihitung ulang; pemeriksaan reproduksi: GB atas EF-L dan GB atas
EF-LD harus sama dengan fase N (empat desimal). Sel: `gb_dc` atas EF-LD, EF-L, kalender, EF-P; `gb_dc` atas `gb`.
Untuk `gb_dc` atas EF-LD dan atas EF-L: batas bawah dua tahap lima benih × 1.000 dan 10.000 ulangan, dan bootstrap
blok (panjang 3, melingkar, 10.000). Uji tanda dan Wilcoxon (12 tahun) untuk `gb_dc` atas EF-LD dan atas EF-L,
deskriptif. Log `DL_FIRE_MODIS/patokan_modis_o_{kab,nas}.log`.

### Aturan keputusan

Rujukan aturan tiap sasaran tetap seperti terdaftar untuk 2026 (D10, `Hasil_Patokan_18.md`): per kabupaten
kalender, EF-P, EF-L; nasional ditambah EF-LD. "Lolos kokoh" = lolos dan batas bawah dua tahap > 0 di kelima benih
bootstrap.

**R45 (VIIRS).** (a) Bila `xgb_nasdc` lolos kokoh atas semua rujukan aturan suatu sasaran, naskah melaporkan hasil
banding positif untuk sasaran itu di §6.2 (tetap eksploratif, dalam-sampel untuk pemilihan rujukan), abstrak
menyebutnya, dan per kabupaten juga dilaporkan apakah ia lolos atas EF-LD. (b) Bila tidak, §6.2 menulis bahwa GB
yang diberi masukan EF-LD tidak terbedakan dari rujukan yang menentukan vonis (dengan angka), dan klausa "which
sees the FWI but not the Drought Code" diganti dengan hasil ini.

**R46 (MODIS maju).** (a) Bila `gb_dc` lolos kokoh atas semua rujukan aturan nasional (kalender, EF-P, EF-L,
EF-LD), kemenangan nasional maju dipulihkan sebagai hasil banding positif eksploratif untuk GB yang diberi DC:
abstrak, §1, §6.3, §7, §9 dan Tabel 5 menyebutnya; judul tetap. Per kabupaten sama dengan rujukan aturan per
kabupaten, dan dilaporkan apakah ia lolos atas EF-LD. (b) Bila tidak, klausa abstrak "which the model was not
given" dan kalimat §6.3 bahwa perbandingan menguji masukan kekeringan diganti: GB yang diberi DC juga tidak
terbedakan dari EF-LD (atau dari rujukan yang menentukan), dengan angka.

**R47 (deskriptif).** `xgb_nasdc` − `xgb_nas` dan `gb_dc` − `gb` dilaporkan di Suplemen (tabel baru S16f) tanpa
klaim; tidak mengubah teks utama kecuali selisihnya lolos kokoh, maka satu kalimat di §6.2 atau §6.3.

Tidak ada rujukan yang ditambah atau dibuang oleh fase ini; rujukan 2026 tidak berubah.

## Hasil fase O (9 Nov 2026)

Log: `DL_FIRE_NASIONAL/penuh/patokan_tingkat_o_{kab,nas}.log`, `DL_FIRE_MODIS/patokan_modis_o_{kab,nas}.log`.
Reproduksi: VIIRS xgb atas EF-LD +0,0498 (kab) dan +0,0044 (nas), EF-L ulang selisih maks 9,9e-17 — sama
dengan fase M. MODIS GB atas EF-L +0,0496 / −0,0615 dan GB atas EF-LD +0,0257 / −0,0972 (nas / kab) — sama
dengan fase N.

**Tabel O1. VIIRS 2020–2025, `xgb_nasdc` (GB + anomali nasional + anomali DC, benih 42).** (a) rata-rata rasio
[dua tahap] t; (b) satu rasio [dua tahap]; (c) median, tahun > 0 dari 6.

| Sel | Per kabupaten | Nasional |
|---|---|---|
| xgb_nasdc atas EF-LD | +0,0676 [−0,0183, +0,1477] t [−0,0417, +0,1769]; (b) +0,0724 [−0,0198, +0,1369]; (c) +0,0879 5/6 | +0,0350 [−0,0411, +0,1128] t [−0,0621, +0,1320]; (b) +0,0247 [−0,0481, +0,1038]; (c) +0,0412 5/6 |
| xgb_nasdc atas EF-L | +0,0845 [+0,0027, +0,1673] t [−0,0246, +0,1937]; (b) +0,0868 [−0,0016, +0,1524]; (c) +0,0986 5/6 | +0,0669 [−0,0055, +0,1486] t [−0,0341, +0,1679]; (b) +0,0538 [−0,0129, +0,1382]; (c) +0,0854 5/6 |
| xgb_nasdc atas kalender | **+0,4207** [+0,2365, +0,6195] t [+0,1427, +0,6988] | **+0,2780** [+0,1549, +0,4324] t [+0,0834, +0,4726] |
| xgb_nasdc atas EF-P | **+0,3913** [+0,2292, +0,5701] t [+0,1502, +0,6323] | **+0,2813** [+0,1585, +0,4342] t [+0,0864, +0,4762] |
| xgb_nasdc atas xgb_nas | −0,0078 [−0,0276, +0,0112] t [−0,0348, +0,0193]; 1/6 | +0,0002 [−0,0125, +0,0144] t [−0,0179, +0,0183]; 3/6 |
| xgb_nasdc atas xgb | +0,0160 [−0,0545, +0,0718] t [−0,0714, +0,1034]; 4/6 | +0,0217 [−0,0184, +0,0758] t [−0,0464, +0,0897]; 3/6 |
| xgb_nas atas EF-LD | +0,0740 [−0,0153, +0,1583] t [−0,0386, +0,1867] | +0,0335 [−0,0542, +0,1197] t [−0,0796, +0,1466] |

Batas bawah dua tahap lima benih + 10.000 ulangan, `xgb_nasdc` atas EF-LD: kab −0,0183..−0,0089, nas
−0,0441..−0,0411; atas EF-L: kab +0,0014..+0,0065 (semua > 0; t95 tidak), nas −0,0151..−0,0055.

**Tabel O2. MODIS maju 2008–2019, `gb_dc` (GB fase N + anomali DC, enam benih).**

| Sel | Per kabupaten | Nasional |
|---|---|---|
| gb_dc atas EF-LD | −0,0476 [−0,1498, +0,0276] t [−0,1467, +0,0515]; (b) −0,0001 [−0,0649, +0,0423]; (c) +0,0229 8/12 | **+0,0380** [+0,0026, +0,0714] t [+0,0035, +0,0725]; (b) +0,0425 [+0,0102, +0,0727]; (c) +0,0602 8/12 |
| gb_dc atas EF-L | −0,0162 [−0,1248, +0,0619] t [−0,1187, +0,0862]; (b) +0,0312 [−0,0380, +0,0750]; (c) +0,0409 8/12 | **+0,0607** [+0,0184, +0,1057] t [+0,0159, +0,1056]; (b) +0,0665 [+0,0290, +0,1069]; (c) +0,0674 10/12 |
| gb_dc atas kalender | **+0,1508** [+0,0783, +0,2324] t [+0,0579, +0,2438]; 11/12 | **+0,1458** [+0,0858, +0,2180] t [+0,0777, +0,2138]; 12/12 |
| gb_dc atas EF-P | **+0,1279** [+0,0596, +0,2028] t [+0,0424, +0,2134]; 10/12 | **+0,1538** [+0,0823, +0,2455] t [+0,0681, +0,2395]; 11/12 |
| gb_dc atas gb | +0,0307 [−0,0110, +0,0839] t [−0,0252, +0,0867]; 8/12 | +0,0119 [−0,0027, +0,0301] t [−0,0062, +0,0301]; 9/12 |

Nasional, batas bawah dua tahap lima benih + 10.000 ulangan: atas EF-LD +0,0000..+0,0047 (semua > 0; benih 2
= +0,0000 pada empat desimal), satu rasio +0,0080..+0,0113; atas EF-L +0,0133..+0,0217; atas kalender
+0,0844..+0,0906; atas EF-P +0,0808..+0,0853. Blok 3: atas EF-LD +0,0380 [+0,0057, +0,0675], satu rasio
[+0,0119, +0,0692]; atas EF-L [+0,0210, +0,0958]. Uji tahunan: atas EF-LD 8/12, tanda p 0,388, Wilcoxon p 0,043;
atas EF-L 10/12, p 0,039, Wilcoxon p 0,021. Per kabupaten: atas EF-LD 8/12, p 0,388, Wilcoxon p 0,791; atas
EF-L 8/12, p 0,388, p 0,569; batas bawah dua tahap atas EF-LD −0,1498..−0,1355, atas EF-L −0,1248..−0,1049; blok 3
atas EF-LD [−0,1502, +0,0301].

### Keputusan

**R45 → (b)** untuk kedua sasaran: `xgb_nasdc` tidak lolos atas EF-LD maupun EF-L (per kabupaten dua tahap atas
EF-L > 0 di kelima benih tetapi t95 mencakup nol). §6.2: GB yang diberi kedua anomali EF-LD tidak terbedakan
dari EF-LD (+0,068 per kabupaten, +0,035 nasional) atau EF-L (+0,085, +0,067); klausa "which sees the FWI but not
the Drought Code" diganti.

**R46 → (a) nasional, (b) per kabupaten.** Nasional `gb_dc` lolos kokoh atas kalender, EF-P, EF-L dan EF-LD
(batas bawah dua tahap atas EF-LD > 0 di kelima benih, paling kecil +0,0000 pada empat desimal; blok 3 juga).
Maka kemenangan nasional maju dipulihkan sebagai hasil banding positif eksploratif untuk GB yang diberi DC:
abstrak, §1, §6.3, §7, §9, Tabel 5; judul tetap. Wajib ditulis bersama: tipisnya batas (nol pada empat desimal
di satu benih), 8 dari 12 tahun (uji tanda p 0,39), dan bahwa model ini ditambahkan sesudah GB tanpa DC kalah
dari EF-LD (pasca-hoc atas hasil fase N). Per kabupaten `gb_dc` tak terbedakan dari EF-LD (−0,048) dan EF-L
(−0,016); klausa "which the model was not given" diganti.

**R47**: `xgb_nasdc` − `xgb_nas` dan `gb_dc` − `gb` tidak lolos di sasaran mana pun → Suplemen S16f saja.

### Penyimpangan bertanggal dari R46a (9 Nov 2026, keputusan penulis)

Telaah internal ketujuh belas (Reject in current form) menilai
kritis (metode, advokat setan, presentasi) bahwa lolosnya `gb_dc` nasional bergantung pada tambahan DC yang sendiri
tak terbedakan dari nol (+0,0119 [−0,0027, +0,0301], t [−0,0062, +0,0301]), dengan batas bawah dua tahap +0,0000
di satu benih, pada model yang ditambahkan sesudah GB tanpa DC gagal atas EF-LD. Penulis 9 Nov memilih
menurunkan klaim: hasil R46(a) tetap dilaporkan di §6.3 dan Tabel 5 dengan angka di atas, tetapi tidak dibaca
sebagai hasil banding positif; abstrak, §1, §7, §8, §9 dan Gambar 3 menulis bahwa tidak ada model yang terbukti
mengalahkan EF-LD nasional, juga bila diberi DC. Dicatat sebagai penyimpangan di Lampiran A naskah.

Ralat rumusan sesudah telaah internal kedelapan belas: menulis
"tak terbedakan dari EF-LD, juga bila diberi DC" salah untuk `gb_dc`, yang kedua selangnya di atas nol. Rumusan
naskah kini: lolos aturan hanya pada batas bawah 0,000 (dekat nol), tidak dibaca sebagai kemenangan. Alasan
penyimpangan diganti: `gb_dc` percobaan kedua pada dua belas tahun yang sama, ditambah sesudah `gb` gagal atas EF-LD,
pada tahun yang sudah dinilai di putaran sebelumnya; tambahan DC +0,012 tak terbedakan dari nol hanya menunjukkan
bahwa GB tanpa DC sudah dekat ambang. Angka tidak berubah.

### Penyimpangan dibatalkan (9 Nov 2026, keputusan penulis sesudah telaah internal kesembilan belas)

Telaah internal kesembilan belas menilai kritis
(advokat setan; metode) bahwa penyimpangan dari R46a menerapkan standar pasca-hoc pada `gb_dc` tetapi tidak pada
EF-LD, yang juga ditambahkan sesudah tahun uji dinilai; GB atas EF-L nasional (batas bawah +0,002) dihitung lolos,
`gb_dc` atas EF-LD (+0,0000..+0,0047) tidak, padahal definisinya sama. Penulis memutuskan mengikuti aturan R46a:
penyimpangan dibatalkan dan R46(a) diikuti seperti terdaftar — kemenangan nasional maju `gb_dc` atas keempat
rujukan dilaporkan sebagai hasil banding positif eksploratif yang tipis, bersama batas tipis, 8/12 tahun (uji
tanda p 0,39) dan urutan penambahan model. Penyimpangan yang ditarik tetap tercatat di Lampiran A. Angka tidak berubah.
