# Hasil Tolok Ukur Nasional 10 — fase E revisi PeatFireBench (rujukan peluang yang sepadan)

**7 November 2026.** Lanjutan `Hasil_Patokan_9.md` (fase D), revisi PeatFireBench. Pemicu: telaah internal
pra-kirim ketiga atas draf revisi (catatan telaah tidak dirilis). Temuan utamanya: keunggulan Brier atas
EF-1m hampir seluruhnya reliabilitas (REL 0,0198 lawan 0,0022; RES 0,0076 lawan 0,0091); sisa sesudah
Platt mungkin hanya model mengikuti tingkat api tahun itu dari data bulan lalu; EF-1m mungkin kalah dari
peluang konstan; rujukan dipilih dengan AUC-PR, tidak pernah dengan kriteria peluang.

## Rancangan fase E (ditulis 7 Nov 2026, SEBELUM skrip mana pun dijalankan)

Data, label, model, dan prediksi fase A–D tidak berubah. Label `y_kabupaten` dan `y_gabungan` tetap
seperti di panel (ambang dari seluruh periode; kebocoran ambang sudah tercatat di fase C, R12).
Populasi evaluasi sama dengan fase D (`ambil_sasaran`: kabupaten dengan positif untuk per kabupaten).
Selang dua tahap seperti fase C/D (tahun lalu kabupaten, 1.000 ulangan, seed 2026) dan selang-t
atas nilai tahunan (df 5). Kelompok tahun: biasa = S1 2020–2025, ekstrem = S2 2015, 2014, 2019.
Satu skrip: `patokan_revisi_peluang.py`, log `DL_FIRE_NASIONAL/penuh/patokan_revisi_peluang.log`.

**Status.** Tahun uji 2020–2025 sudah dipakai di fase A–D. Semua hasil fase E eksploratori; tidak
ada kontras E yang masuk himpunan konfirmatori D2, dan naskah menyatakannya.

### E1. Rujukan peluang baru, dari tahun latih lipatan saja

Untuk tiap lipatan (S1 tahun uji t: latih tahun < t; S2: latih semua tahun kecuali t) dan tiap sasaran,
dihitung dari panel (`panel_nasional_fitur_penuh.csv`) pada populasi yang sama dengan evaluasi:

- `konstan` — laju positif di baris latih.
- `konstan_bulan` — laju positif per bulan kalender di baris latih (musim nasional).
- `ef1` — EF-1m dihitung ulang (bagian tahun latih yang positif untuk kabupaten dan bulan kalender
  itu); diperiksa sama dengan `klim_frek1` tersimpan (selisih maks dicetak, tanpa pemecah seri).
- `ef3` — sama dengan jendela tiga bulan (EF-3m).
- `ef1_pasca` — EF-1m dari tahun latih ≥ 2016 saja (S1 2020: empat tahun; S2: sepuluh tahun).
- `ef1_halus` — (s + κ·ef3) / (n + κ), κ = 2 tetap (s, n = jumlah positif dan banyak tahun latih untuk
  kabupaten-bulan itu). κ tidak disetel.

**Pemilihan rujukan peluang (sejajar aturan AUC-PR §4.2).** Per lipatan dan sasaran, dari enam calon
di atas dipilih yang Brier-nya terendah dalam validasi tinggalkan-satu-tahun di tahun latih (tiap tahun
latih v dinilai dengan rujukan yang dihitung dari tahun latih selain v). Hasilnya `rujukan_peluang`.
Pilihan per lipatan dicetak.

**Kalender disesuaikan tingkat tahun (`ef1_tingkat`).** Regresi logistik pada baris latih:
logit p = a + b·logit(ef1_halus) + c·A, dengan A = log1p(jumlah titik panas nasional bulan lalu) −
rata-rata nilai itu di tahun latih untuk bulan kalender yang sama. Untuk baris latih, `ef1_halus` dan
rata-rata A dihitung tanpa tahun baris itu sendiri. Rujukan ini memakai data bulan lalu tingkat
nasional saja, tanpa riwayat kabupaten; ia menjawab apakah keunggulan peluang model lebih dari sekadar
mengikuti tingkat api tahun itu.

### E2. Pemecahan Brier: tingkat tahun lawan sisanya (diagnostik oracle)

Untuk tiap tahun uji dan tiap ramalan f, δ dicari sehingga rata-rata sigmoid(logit f + δ) = laju
positif teramati tahun itu (pencarian akar, logit dipotong di 1e-6). Brier ramalan yang digeser ini
= Brier "tingkat benar". δ dihitung sekali per tahun pada seluruh baris tahun itu, lalu tetap selama
bootstrap (dicatat: selang sedikit lebih sempit dari bila δ dihitung ulang). Ini memakai label uji dan
bukan ramalan; ia hanya memisahkan bagian keunggulan yang hilang bila semua ramalan diberi tingkat
tahun yang benar. Dilaporkan untuk xgb, rf, mlp, `log_ef1_jeda`, `xgb_setel` atas EF-1m/EF-3m dan atas
`rujukan_peluang`: BSS mentah, BSS tingkat-benar, dan bagian tingkat = 1 − BSS tingkat-benar / BSS
mentah, dengan selang dua tahap dan t (tahun biasa) atau per tahun (ekstrem).

### E3. BSS untuk semua model yang memberi peluang

Semua kolom prediksi yang nilainya di [0, 1] pada semua baris (dari `patokan_revisi_prediksi.csv`,
`patokan_revisi_sederhana.csv`, `patokan_revisi_fdrs.csv`, `patokan_revisi_setel.csv`,
`patokan_fwi_adil.csv`); kolom yang dikeluarkan dicetak beserta alasannya. Tiap kolom dinilai dengan
BSS atas (a) rujukan pasangan Tabel 2 (EF-1m / EF-3m), (b) `rujukan_peluang`, (c) `ef1_tingkat`, kedua
sasaran, kedua kelompok tahun, dengan selang dua tahap dan t untuk tahun biasa.

### E4. Pemotongan log score

Log score dihitung dengan ε = 1e-4 (seperti fase D) dan ε = 1e-3. Dicetak: bagian positif yang
ramalan EF-1m-nya < 1e-3, dan selisih log score xgb/rf atas EF-1m dan atas `rujukan_peluang` pada
kedua ε.

### Aturan putusan (ditetapkan sekarang)

- **R24.** Bila `konstan` atau `konstan_bulan` lebih baik dari EF-1m dalam Brier per kabupaten (BSS atas
  EF-1m > 0, lolos selang dua tahap dan t) → naskah menyatakan EF-1m lebih buruk daripada peluang
  konstan, dan BSS atas EF-1m tidak lagi dipakai di judul atau abstrak; klaim peluang hanya atas
  `rujukan_peluang`.
- **R25.** "Better as a probability" (atau padanannya) tetap di judul/abstrak hanya bila BSS xgb DAN
  rf atas `rujukan_peluang` per kabupaten lolos kedua selang (batas bawah dua tahap dan t > 0). Bila
  hanya nasional → disebut dengan sasarannya. Bila tidak satu pun → separuh peluang dilaporkan tak
  terputuskan seperti urutan.
- **R26.** Bila BSS xgb atas `ef1_tingkat` per kabupaten tidak lolos kedua selang, ATAU BSS tingkat-
  benar xgb atas EF-1m per kabupaten < sepertiga BSS mentahnya atau selang dua tahapnya memuat nol →
  naskah menulis keunggulan peluang sebagai terutama mengikuti tingkat api tahun itu dari data bulan
  lalu, dan tidak memberinya kepada "machine learning" melebihi itu.
- **R27.** Kata "every model" / "all models" untuk BSS hanya bila semua kolom E3 positif dan lolos
  kedua selang atas rujukan yang disebut; selain itu naskah memberi hitungan "k dari n".
- **R28.** Bila rujukan peluang terpilih berbeda antarlipatan, naskah menyebut pilihan per lipatan
  (tabel suplemen) dan tidak menamai satu rujukan sebagai "the" probability reference.

### D10 diperluas (pendaftaran saja)

Evaluasi prospektif 2026 (D10) juga melaporkan BSS atas `rujukan_peluang` dan `ef1_tingkat` yang
dihitung dengan aturan E1 dari tahun latih 2012–2025.

### Tidak dikerjakan di fase E

Penyetelan κ; rujukan tingkat tahun tingkat provinsi; ambang dari tahun latih saja untuk label (R12
tetap tercatat sebagai batasan); perbaikan teks dari telaah ketiga yang tidak memerlukan analisis
(judul, R16 di abstrak, istilah "nationally", tabel) dikerjakan di naskah sesudah hasil E.

## Hasil (dijalankan 7 Nov 2026; komit rancangan `989073c`)

Log: `DL_FIRE_NASIONAL/penuh/patokan_revisi_peluang_kab.log` (per kabupaten) dan `…_nas.log` (nasional).
Prediksi rujukan: `patokan_revisi_peluang_y_kabupaten.csv`, `…_y_gabungan.csv` (tidak masuk repo).

**Penyimpangan teknis (bukan analisis).** (1) Jalan pertama (satu proses, dua sasaran) dihentikan sistem
karena memori laptop; skrip diberi argumen sasaran dan dijalankan dua kali, satu per sasaran, dengan
benih dan isi yang sama. (2) Dua galat kode diperbaiki sebelum jalan yang dilaporkan: penyaring baris E3
mencampur S1 2019 dengan S2 2019, dan batas [0, 1] menolak `klim_frek1` karena pemecah seri 1e-6.
(3) Baris "Ringkasan R27" di akhir log salah hitung (jumlah kolom objek); hitungan yang benar adalah baris
"ringkas" di dalam tiap bagian, dan itu yang dipakai di bawah. (4) Aturan E3 "nilai di [0, 1]" memasukkan
tiga skor persentil yang bukan peluang (`dc_persentil`, `fwi_persentil`, `fwi_kini_persentil`); BSS-nya
sangat negatif (−4 sampai −9) dan dilaporkan apa adanya, tetapi naskah menghitung model peluang tanpa ketiganya.

### E1. Rujukan peluang terpilih

Cek: `ef1`/`ef3` dihitung ulang sama dengan `klim_frek1`/`klim_frek` tersimpan (selisih maks 1e-6 =
pemecah seri). Pilihan tinggalkan-satu-tahun di tahun latih: per kabupaten S1 2020/22/23 `ef3`, 2021/24/25
`ef1_halus`, S2 semua `ef1_halus`; nasional S1 2020–23 `ef3`, 2024–25 `ef1_halus`, S2 semua `ef1_halus`.
`konstan`, `konstan_bulan`, dan `ef1_pasca` tidak pernah terpilih (Brier latih `konstan` 0,080–0,110 lawan
`ef1` 0,047–0,088). `ef1_tingkat`: koefisien logit ef1_halus +0,54 sampai +0,78, koefisien anomali api
nasional bulan lalu +0,71 sampai +1,11 di semua 18 lipatan.

Brier rata-rata tahun uji biasa:

| Sasaran | konstan | konstan_bulan | EF-1m | EF-3m | ef1_pasca | ef1_halus | terpilih | ef1_tingkat | GB | RF |
|---|---|---|---|---|---|---|---|---|---|---|
| Per kab. | 0,0423 | 0,0441 | 0,0496 | 0,0440 | 0,0427 | 0,0476 | 0,0462 | 0,0321 | 0,0297 | 0,0303 |
| Nasional | 0,0600 | 0,0578 | 0,0451 | 0,0416 | 0,0395 | 0,0437 | 0,0417 | 0,0317 | 0,0311 | 0,0303 |

### R24. Kalender lawan peluang konstan

Per kabupaten, tahun biasa, BSS `konstan` atas EF-1m **+0,189** [2 tahap −0,126, +0,449] [t −0,233, +0,612];
per tahun +0,319 / +0,514 / +0,576 / −0,522 / +0,015 / +0,233 (konstan lebih baik di 5 dari 6 tahun, kalah
jauh di 2023). `konstan_bulan` +0,126 [−0,025, +0,267]; `ef1_pasca` +0,162 [+0,017, +0,270] [t −0,025,
+0,348]; `ef1_halus` +0,041 [+0,020, +0,061] [t +0,012, +0,071] (lolos). Tahun ekstrem per kabupaten: konstan
jauh lebih buruk (−0,32 / −0,35 / −0,44). Nasional: konstan atas EF-3m −0,453 [−0,827, −0,076].

**Putusan R24: tidak terpicu** (konstan tidak lolos kedua selang). Tetapi titiknya positif dan Brier EF-1m
(0,0496) di atas konstan (0,0423) di tahun biasa; naskah wajib menyebutnya: di tahun biasa EF-1m mentah tidak
lebih baik daripada peluang konstan, sehingga BSS atas EF-1m bukan ukuran keterampilan yang berarti.

### R25. Model atas rujukan peluang terpilih

| Sasaran, tahun biasa | GB | RF | GB tersetel | Logistik EF-1m + jeda |
|---|---|---|---|---|
| Per kab. | +0,383 [+0,216, +0,561] [t +0,147, +0,619] | +0,369 [+0,215, +0,527] [t +0,152, +0,586] | +0,376 [+0,192, +0,570] | +0,277 [+0,143, +0,417] |
| Nasional | +0,259 [+0,124, +0,428] [t +0,039, +0,478] | +0,274 [+0,151, +0,425] [t +0,079, +0,470] | +0,283 [+0,162, +0,435] | +0,251 [+0,149, +0,359] |

Tahun ekstrem per kabupaten: GB +0,405 / −0,026 / +0,283, RF +0,302 / −0,067 / +0,220 (2015 / 2014 / 2019).
**Putusan R25:** syarat terpenuhi untuk kedua sasaran (GB dan RF lolos kedua selang atas rujukan terpilih).

### R26. Kalender disesuaikan tingkat tahun, dan pemecahan oracle

BSS atas `ef1_tingkat` (kalender + anomali api nasional bulan lalu, tiga parameter), tahun biasa:

| Sasaran | GB | RF | GB tersetel | MLP | Logistik EF-1m + jeda | lolos kedua selang |
|---|---|---|---|---|---|---|
| Per kab. | +0,067 [−0,020, +0,152] [t −0,048, +0,181] | +0,030 [−0,055, +0,102] | +0,065 [−0,051, +0,175] | −0,126 [−0,346, +0,067] | −0,147 [−0,303, −0,005] | 0 dari 22 (titik > 0: 4) |
| Nasional | +0,037 [−0,094, +0,152] | +0,058 [−0,000, +0,127] [t −0,023, +0,138] | +0,069 [−0,018, +0,160] | −0,051 | +0,012 | 0 dari 22 (titik > 0: 7) |

`ef1_tingkat` sendiri atas EF-1m per kabupaten +0,376 [+0,170, +0,574] [t +0,083, +0,669]; atas EF-3m nasional
+0,232 [+0,113, +0,363] [t +0,061, +0,402]. Tahun ekstrem, GB atas `ef1_tingkat`: per kab. +0,145 / −0,063 /
+0,197, nasional +0,178 / +0,078 / +0,209; RF nasional +0,171 / +0,068 / +0,186.

Pemecahan oracle (E2), tahun biasa, atas rujukan pasangan:

| Sasaran | Model | BSS mentah | BSS tingkat-benar [2 tahap] [t] | Bagian tingkat [2 tahap] |
|---|---|---|---|---|
| Per kab. | GB | +0,415 | +0,054 [−0,002, +0,107] [−0,012, +0,120] | 0,87 [0,69, 1,01] |
| Per kab. | RF | +0,401 | +0,048 [+0,004, +0,087] [−0,001, +0,097] | 0,88 [0,73, 0,99] |
| Per kab. | GB tersetel | +0,409 | +0,071 [+0,001, +0,132] [−0,010, +0,152] | 0,83 [0,58, 1,00] |
| Per kab. | Logistik EF-1m + jeda | +0,322 | +0,020 [−0,042, +0,071] | 0,94 [0,77, 1,22] |
| Nasional | GB | +0,255 | +0,098 [+0,062, +0,134] [+0,070, +0,126] | 0,62 [0,22, 0,79] |
| Nasional | RF | +0,271 | +0,079 [+0,049, +0,111] [+0,056, +0,101] | 0,71 [0,38, 0,86] |
| Nasional | GB tersetel | +0,279 | +0,113 [+0,075, +0,153] [+0,081, +0,146] | 0,59 [0,22, 0,78] |
| Nasional | Logistik EF-1m + jeda | +0,248 | +0,085 [+0,036, +0,132] [+0,030, +0,140] | 0,66 [0,42, 0,84] |

Tahun ekstrem, GB tingkat-benar atas EF-1m per kab. +0,272 / +0,018 / +0,159; nasional +0,208 / +0,073 / +0,185.

**Putusan R26: terpicu untuk kedua sasaran.** Per kabupaten kedua syarat terpenuhi (GB atas `ef1_tingkat`
tidak lolos; BSS tingkat-benar +0,054 < sepertiga dari +0,415 dan selang dua tahapnya memuat nol). Nasional
syarat pertama terpenuhi (tidak satu model pun lolos atas `ef1_tingkat`), sedangkan syarat kedua tidak: sesudah
tingkat tahun dibuat benar, GB/RF/GB tersetel/logistik masih unggul +0,08 sampai +0,11 dengan kedua selang di
atas nol (diagnostik oracle, eksploratori). Jadi: sekitar 87 % keunggulan Brier per kabupaten dan 60–70 %
nasional adalah tingkat api tahun itu, dan sebuah kalender yang diberi jumlah titik panas nasional bulan lalu
menangkapnya; di atas kalender itu tidak satu pun dari 22 kolom terbedakan di tahun biasa.

### R27. "Every model"

Atas rujukan terpilih, tahun biasa: per kabupaten 14 dari 22 kolom lolos kedua selang (14 dari 19 model
peluang, tanpa tiga skor persentil), titik > 0 pada 17; nasional 8 dari 22 (8 dari 19), titik > 0 pada 13
(logistik bulanan, MLP, dan beberapa baseline cuaca tidak). Atas EF-1m per kabupaten 12 dari 21.
**Putusan R27:** "every model" dilarang; naskah memberi hitungan.

### R28. Pilihan rujukan antarlipatan

Terpilih `ef3` atau `ef1_halus`, berganti antarlipatan, jadi **R28 terpicu**: naskah menyebut pilihan per
lipatan (tabel suplemen) dan menamai rujukannya "the training-selected probability climatology", bukan satu
klimatologi tertentu. Catatan: di tahun uji `ef1_pasca` lebih baik daripada pilihan (Brier 0,0427 lawan
0,0462 per kab.; 0,0395 lawan 0,0417 nasional), sejalan dengan rezim pasca-2015 (D8); aturan pemilihan dari
tahun latih tidak menangkapnya.

### E4. Pemotongan log score

Bagian positif dengan EF-1m < 1e-3: 9,1 % (tahun biasa), 14,5 % (ekstrem); rujukan terpilih 2,1 % / 3,7 %;
nasional EF-3m 1,4 % / 3,6 %. Selisih log score GB atas EF-1m per kabupaten +0,073 (ε 1e-4) dan +0,065
(ε 1e-3), t95 di atas nol pada keduanya; atas rujukan terpilih +0,055 / +0,054; atas `ef1_tingkat` +0,012
[t −0,001, +0,024] pada kedua ε. Nasional GB atas `ef1_tingkat` +0,008 [−0,006, +0,022]. Pemotongan tidak
mengubah putusan.

## Putusan fase E dan akibat untuk naskah

1. **Separuh "better as a probability" gugur sebagai klaim tentang pembelajaran mesin.** Model memang memberi
   Brier jauh lebih baik daripada kalender mentah dan daripada klimatologi peluang terpilih (R25), tetapi hampir
   semuanya (87 % per kab.) adalah mengikuti tingkat api tahun itu, dan kalender yang diberi satu angka —
   anomali titik panas nasional bulan lalu — menutup selisih itu sampai tidak terbedakan untuk 22 dari 22
   kolom (R26). Judul "better as a probability" dilarang.
2. **Tesis baru (untuk dirumuskan di naskah):** di panel ini, baik sebagai urutan maupun sebagai peluang,
   model yang diuji tidak terbedakan dari kalender kabupaten itu sendiri yang disesuaikan dengan jumlah titik
   panas nasional bulan lalu. Keterampilan atas kalender mentah sebagian besar adalah tingkat tahun, bukan
   pembedaan antarkabupaten. Rujukan yang harus dikalahkan karya berikutnya: kalender + tingkat tahun.
   Ini menjadikan sumbangan utama tolok ukur + protokol + rujukan itu (saran telaah internal ketiga).
3. **Yang boleh disebut eksploratori:** nasional, sesudah tingkat dibuat benar (oracle), model masih
   memisahkan kabupaten-bulan lebih baik (+0,08–0,11); di tahun ekstrem GB unggul atas `ef1_tingkat` di
   2015 dan 2019 (bukan 2014). Keduanya bukan klaim utama.
4. **EF-1m mentah sebagai rujukan peluang** tidak dipakai lagi di judul/abstrak; naskah menyebut bahwa di
   tahun biasa ia tidak lebih baik daripada peluang konstan (titik; R24 tidak terpicu).
5. **"Every model"** diganti hitungan (R27); pilihan rujukan per lipatan ke suplemen (R28).
6. **Belum diputuskan:** penulisan ulang naskah keempat kalinya, telaah keempat, dan apakah fase E ini
   cukup untuk berhenti (penulis).
