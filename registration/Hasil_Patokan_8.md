# Hasil Tolok Ukur Nasional 8 — fase C revisi PeatFireBench (ketidakpastian antartahun, rujukan per lipatan, siaga, kebocoran ambang)

**6 November 2026.** Lanjutan `Hasil_Patokan_7.md` (fase B), revisi PeatFireBench. Pemicu: telaah internal
pra-kirim atas draf revisi (catatan telaah tidak dirilis).

## Rancangan fase C (ditulis 6 Nov 2026, SEBELUM skrip mana pun dijalankan)

Data, model, dan prediksi tidak berubah kecuali disebut. Benih 42 (sama dengan Tabel 2 naskah, yang
memakai kolom benih 42 dari `patokan_revisi_prediksi.csv`); rata-rata enam benih dicetak sebagai baris
kepekaan bila tersedia (S1 saja). Penamaan rujukan di bawah dan di naskah: **EF-1m** = `klim_frek1`
(frekuensi lampau bulan kalender itu), **EF-3m** = `klim_frek` (jendela tiga bulan), **MC** =
`klimatologi` (rata-rata hitungan). Pasangan Tabel 2 = EF-1m untuk per kabupaten, EF-3m untuk nasional.

**Selang dua tahap** (dipakai di seluruh fase C): tiap ulangan mengambil ulang tahun uji dengan
pengembalian dari tahun dalam kelompoknya, lalu kabupaten dengan pengembalian (tarikan kabupaten sama
untuk semua tahun dalam ulangan itu); statistik = rata-rata atas tahun terambil. Selang kabupaten =
tahun tetap, kabupaten diambil ulang. Seed 2026.

### C1. Ketidakpastian antartahun untuk Tabel 2 — `patokan_revisi_tahun.py`

S1 2020–2025, kedua sasaran, semua 15 baris Tabel 2 (FWI persentil dan klimatologi + FWI dari
`patokan_fwi_adil.csv`). Untuk tiap model terhadap rujukan pasangan Tabel 2: keunggulan dalam-tahun,
selang kabupaten (1.000), selang dua tahap (1.000), keunggulan per tahun 2020–2025, banyak tahun
positif, dan statistik t atas enam tahun (rata-rata/simpangan baku × √6, df 5). Per kabupaten juga
keunggulan atas EF-3m (kolom tambahan Tabel 2). "Inflasi" diganti selisih keunggulan: keunggulan atas
MC − keunggulan atas rujukan = AP(rujukan) − AP(MC), sama untuk semua model; dilaporkan dengan selang
kabupaten dan dua tahap. Rasio lama (4,9 dan 6,5) dicetak per model (rf, xgb, mlp) dengan selang
persentil dua tahap, hanya untuk dicatat. Baris kepekaan: rf, xgb, mlp rata-rata enam benih.

- **R8.** Abstrak dan kesimpulan boleh menyatakan model menambah keterampilan atas rujukan untuk suatu
  sasaran hanya untuk model yang batas bawah **selang dua tahap**-nya > 0. Bila tidak ada model yang
  lolos untuk per kabupaten, naskah menulis bahwa keunggulan per kabupaten tidak dapat dibedakan dari nol
  bila variasi antartahun diperhitungkan, dan menyebut tahun-tahun negatif.

### C2. Kontras tahun ekstrem pada pasangan Tabel 2 — `patokan_revisi_tahun.py` (skrip sama)

Kontras ekstrem (S2 2015, 2014, 2019) − tenang (S1 2020, 2021, 2022, 2024, 2025) keunggulan gradient
boosting, random forest, MLP; per kabupaten atas EF-1m, nasional atas EF-3m; selisih antar-sasaran;
m1 (AUC-PR) dan m4 (ROC-AUC); selang kabupaten 300 dan dua tahap 1.000 (tahun ekstrem diambil ulang
dari tiga, tenang dari lima). Kepekaan: tenang termasuk 2023.

- **R9.** Naskah menulis kontras ini apa adanya bersama nilai atas MC (+0,293), EF-3m (−0,010) dan EF-1m
  (+0,070). Bila selang dua tahap m1 pada pasangan Tabel 2 mencakup nol, kalimatnya: pola tahun ekstrem
  tidak dapat dibedakan dari nol pada rujukan benchmark dan tandanya bergantung pada rujukan. Bila tidak
  mencakup nol, naskah menyebut tanda dan besarnya, dan bahwa ROC-AUC (m4) berlawanan tanda bila memang
  demikian. Kata "vanishes" tidak dipakai dalam kedua kasus.

### C3. Rujukan dipilih di dalam lipatan — `patokan_revisi_rujukan.py`

Untuk tiap lipatan luar (S1 2020–2025: latih tahun < uji; S2 2014, 2015, 2019: latih semua tahun lain)
dan kedua sasaran: tiap calon (MC, EF-3m, EF-1m) dihitung luar-lipatan pada tiap tahun latih v dari
tahun latih tanpa v (fungsi `patokan_revisi_prediksi.py`); AUC-PR dalam-tahun dirata-rata atas v (per
kabupaten: kabupaten `ada_positif`). Calon tertinggi = rujukan lipatan itu. Dilaporkan: pilihan per
lipatan; keunggulan semua baris Tabel 2 atas rujukan per lipatan, selang kabupaten dan dua tahap; S2 per
tahun.

- **R10.** Bila rujukan per lipatan sama dengan pasangan Tabel 2 di keenam lipatan S1 untuk suatu
  sasaran, Tabel 2 tetap dan naskah menyatakan bahwa aturan per lipatan memilih rujukan yang sama. Bila
  berbeda di satu lipatan atau lebih, keunggulan atas rujukan per lipatan menjadi kolom utama Tabel 2 dan
  pasangan lama menjadi baris kepekaan. Aturan per lipatan ditulis sebagai rujukan tetap benchmark untuk
  penulis mendatang.

### C4. Siaga dan nilai keputusan atas EF-1m — `patokan_revisi_siaga.py`

Per kabupaten, kelompok biasa (S1 2020–2025) dan ekstrem (S2 2014, 2015, 2019), seperti A4–A5:

- Kuota bulanan (10 % kabupaten tertinggi tiap bulan): hit rate dan FAR untuk xgb, rf, mlp,
  `log_klim_jeda`, EF-1m, EF-3m, MC; selisih hit rate xgb − EF-1m dan rf − EF-1m dengan selang
  kabupaten (500) dan dua tahap (1.000). Juga untuk sasaran nasional (xgb, rf, EF-3m, EF-1m, MC).
- Ambang 10 % luar-lipatan (aturan a): xgb, EF-1m, EF-3m, MC, `log_klim_jeda` (prakiraan luar-lipatan
  dari `patokan_revisi_oof.csv`).
- Nilai ekonomi relatif, a ∈ {0,05; 0,1; 0,2; 0,3; 0,5}: (i) "siaga bila peluang > a" untuk xgb, EF-1m,
  EF-3m (ketiganya peluang) — **perbandingan utama yang dapat dicapai**; (ii) ambang luar-lipatan;
  (iii) amplop. Selisih xgb − EF-1m untuk (i) dan (ii) dengan selang kabupaten (500) dan dua tahap (500).

- **R11.** Abstrak memakai angka kuota atas EF-1m. "Menangkap lebih banyak bulan parah daripada
  kalender kabupaten sendiri" boleh ditulis untuk kelompok tahun yang selang dua tahap selisih hit rate
  xgb − EF-1m-nya > 0; bila mencakup nol, naskah menyebut angka tanpa klaim itu. Nilai ekonomi: "lebih
  bernilai pada rasio a" hanya untuk a yang batas bawah selang dua tahap (i) > 0.

### C5. Kebocoran ambang terhadap rujukan sepadan — `patokan_revisi_ambang.py`

Ambang sasaran dihitung ulang dari tahun latih tiap lipatan (fungsi `label_ulang` dari
`patokan_kepekaan_ambang.py`, panel `panel_nasional_bersih.csv`); prediksi rf, xgb, mlp dengan label
ulang diambil dari `patokan_prediksi_kepekaan_penuh.csv` (benih 42, fitur penuh). EF-1m dan EF-3m
dihitung dari label ulang tahun latih. Dua himpunan kabupaten per kabupaten: (a) tetap (`ada_positif`
baku, seperti uji lama); (b) per lipatan: kabupaten dengan ≥ 1 bulan positif di tahun latih lipatan itu
menurut label ulang. Dilaporkan: Tabel 2 (rf, xgb, mlp atas pasangan Tabel 2) dengan label baku dan
dengan label ulang, berdampingan, selang kabupaten (1.000) dan dua tahap (1.000); S2 per tahun.

- **R12.** Bila label ulang mengubah keunggulan suatu model atas rujukan lebih dari setengah nilai
  label bakunya, atau membalik tanda batas bawah selang dua tahapnya, hasilnya masuk badan naskah (bagian
  6.1) sebagai batasan. Selain itu, satu kalimat di Limitations dengan angka.

### C6. Baseline sederhana yang lebih kuat — `patokan_revisi_sederhana.py`

Dua baseline baru, S1 2020–2025 dan S2, kedua sasaran, benih 42, dari tahun latih saja:

- `log_ef1_jeda`: regresi logistik terskala pada logit(EF-1m), y bulan lalu (sasaran yang sama), log1p
  tp_lag1..3, bulan_sin, bulan_cos. Untuk baris latih, EF-1m dihitung tanpa tahun baris itu sendiri;
  logit dipotong di [10⁻⁴, 1 − 10⁻⁴].
- `xgb_sejarah`: gradient boosting (parameter sama dengan `patokan_prediksi.py`) hanya pada tp_lag1,
  tp_lag2, tp_lag3, tp_lag12, bulan_sin, bulan_cos.

Dilaporkan: AUC-PR dalam-tahun, keunggulan atas rujukan pasangan Tabel 2, dan selisih xgb (penuh) −
baseline baru dan rf − baseline baru, selang kabupaten (1.000) dan dua tahap (1.000); S2 per tahun.

- **R13.** Bila selang dua tahap xgb − `log_ef1_jeda` atau xgb − `xgb_sejarah` mencakup nol untuk suatu
  sasaran, klaim model untuk sasaran itu dipersempit menjadi "menambahkan sejarah api bulan lalu ke
  kalender kabupaten" dan baseline itu masuk Tabel 2 sebagai pembanding utama kedua.

### C7. Kalibrasi ulang hanya pada tahun sesudah 2015 — `patokan_revisi_pasca2015.py`

Gradient boosting, per kabupaten, seperti B2: Platt pada prakiraan luar-lipatan tahun latih, tetapi hanya
tahun dalam ≥ 2016; dibandingkan dengan Platt semua tahun dan mentah. S1 2020–2025: ECE (10 bin lebar
sama), rata-rata peluang dan prevalensi teramati per tahun; S2 per tahun.

- **R14.** Bila ECE rata-rata tahun biasa dengan Platt pasca-2015 < ECE mentah (0,023) **dan** selisih
  |rata-rata peluang − prevalensi| mengecil pada ≥ 4 dari 6 tahun biasa, naskah menulis bahwa
  salah-prakiraan tahun tenang konsisten dengan pergeseran rezim sesudah 2015. Bila tidak, naskah menulis
  bahwa kalibrasi ulang pada tahun sesudah 2015 juga tidak menghilangkannya.

### Tidak dikerjakan di fase C (dicatat sekarang)

- Latensi data operasional (VIIRS NRT, CHIRPS awal): arsip NRT tidak tersedia di laptop; naskah
  memindahkannya ke cakupan abstrak dan Limitations sebagai syarat hasil.
- Ambang per lipatan sebagai bawaan benchmark: hanya uji kepekaan (C5).
- Peta kabupaten yang disiagakan kuota: menunggu keputusan bingkai sesudah C4.

Pengecekan literatur (rantai keputusan karhutla, produk BMKG, kebijakan pasca-2015, rujukan ProbFire,
status Kieft 2026) dikerjakan terpisah dan dicatat di `Rujukan_Revisi_PeatFireBench.md`.

## Hasil (dijalankan 6 Nov 2026; komit rancangan `ad60838`)

Log di `DL_FIRE_NASIONAL/penuh/patokan_revisi_{tahun,rujukan,siaga,ambang,sederhana,pasca2015}.log`
(tidak masuk repo). `ap_cepat` diperiksa sama dengan sklearn (selisih < 10⁻¹²) di awal tiap skrip.

### C1. Tabel 2 dengan ketidakpastian antartahun

Keunggulan dalam-tahun atas rujukan pasangan Tabel 2, S1 2020–2025, benih 42 [selang kabupaten 1.000]
[dua tahap 1.000]; t atas enam tahun (df 5); n+ = tahun positif dari enam.

| Model | Per kab atas EF-1m | dua tahap | t | n+ | per tahun 2020 … 2025 | Per kab atas EF-3m [kab] [dua tahap] |
|---|---|---|---|---|---|---|
| RF | +0,043 [+0,016, +0,071] | [−0,013, +0,098] | 1,90 | 5 | +0,094 +0,035 +0,047 +0,114 −0,035 +0,004 | +0,095 [+0,075, +0,122] [+0,053, +0,153] |
| GB | +0,031 [−0,002, +0,066] | [−0,031, +0,101] | 1,20 | 4 | +0,034 +0,024 +0,115 +0,088 −0,059 −0,015 | +0,083 [+0,060, +0,115] [+0,038, +0,144] |
| MLP | +0,012 [−0,021, +0,042] | [−0,044, +0,076] | 0,60 | 4 | −0,040 +0,010 +0,096 +0,031 −0,033 +0,009 | +0,064 [+0,039, +0,091] [+0,021, +0,121] |
| Local history | −0,044 [−0,070, −0,021] | [−0,096, −0,001] | −2,18 | 1 | | +0,008 [−0,007, +0,026] [−0,028, +0,048] |
| EF-3m | −0,052 [−0,073, −0,035] | [−0,100, −0,013] | −2,65 | 1 | −0,095 −0,025 −0,023 +0,013 −0,070 −0,110 | — |
| MC | −0,119 [−0,146, −0,098] | [−0,181, −0,065] | −4,09 | 0 | | −0,068 [−0,083, −0,055] [−0,136, −0,016] |
| RF 6 benih | +0,043 [+0,013, +0,072] | [−0,016, +0,102] | 1,81 | 5 | | |
| GB 6 benih | +0,036 [+0,000, +0,069] | [−0,029, +0,107] | 1,31 | 4 | | |
| MLP 6 benih | +0,011 [−0,019, +0,039] | [−0,041, +0,063] | 0,58 | 3 | | |

| Model | Nasional atas EF-3m | dua tahap | t | n+ | per tahun 2020 … 2025 |
|---|---|---|---|---|---|
| RF | +0,050 [+0,032, +0,067] | [+0,019, +0,084] | 3,66 | 5 | +0,087 +0,065 +0,080 +0,032 −0,002 +0,038 |
| GB | +0,038 [+0,021, +0,054] | [+0,005, +0,074] | 2,51 | 5 | +0,070 +0,046 +0,081 +0,013 −0,020 +0,039 |
| MLP | +0,026 [+0,007, +0,044] | [−0,006, +0,058] | 1,89 | 4 | −0,011 +0,060 +0,055 +0,015 −0,014 +0,049 |
| Local history | −0,014 [−0,035, +0,007] | [−0,051, +0,022] | −0,91 | 2 | +0,034 −0,024 −0,030 +0,029 −0,060 −0,032 |
| EF-1m | −0,023 [−0,031, −0,012] | [−0,043, −0,004] | −2,76 | 0 | |
| MC | −0,209 [−0,240, −0,169] | [−0,267, −0,141] | −7,01 | 0 | |
| RF / GB / MLP 6 benih | +0,051 / +0,041 / +0,041 | [+0,021, +0,083] / [+0,008, +0,078] / [+0,011, +0,074] | 3,92 / 2,65 / 3,34 | 6 / 5 / 6 | |

Semua baseline lain di bawah rujukan dengan selang dua tahap di bawah nol, kecuali per kabupaten atas
EF-3m (lihat log). Selisih keunggulan (atas MC − atas rujukan) = AP(rujukan) − AP(MC), sama untuk semua
model: per kabupaten +0,119 [+0,098, +0,146], dua tahap [+0,065, +0,181] (per tahun +0,129 +0,039
+0,026 +0,190 +0,185 +0,147); nasional +0,209 [+0,169, +0,240], dua tahap [+0,141, +0,267]. Rasio lama
(dicatat saja): per kabupaten RF 3,77, GB 4,80, MLP 10,92, persentil dua tahap tak terbatas (penyebut
≤ 0 pada 5 %, 16 %, 37 % ulangan); nasional RF 5,17 [3,39, 10,73], GB 6,46 [3,52, 21,07], MLP 9,10.

**R8.** Per kabupaten: tidak ada model dengan batas bawah dua tahap > 0 (RF −0,013, GB −0,031; enam benih
sama) — naskah menulis bahwa keunggulan per kabupaten atas EF-1m tidak dapat dibedakan dari nol bila
variasi antartahun diperhitungkan, dan menyebut 2024 dan 2025 (negatif untuk GB; RF −0,035 di 2024).
Nasional: RF dan GB lolos (+0,050 [+0,019, +0,084], +0,038 [+0,005, +0,074]); MLP benih 42 tidak (enam
benih lolos). Atas EF-3m, per kabupaten ketiga model lolos dua tahap.

### C2. Kontras tahun ekstrem pada pasangan Tabel 2 (411 kabupaten, kedua sasaran, seperti A2)

Kontras ekstrem (S2 2015, 2014, 2019) − tenang (S1 2020, 2021, 2022, 2024, 2025), selang kabupaten 300,
dua tahap 1.000.

| Model | Ukuran | Per kab (atas EF-1m) | Nasional (atas EF-3m) | Selisih antar-sasaran | dua tahap |
|---|---|---|---|---|---|
| GB | m1 AUC-PR | +0,022 [−0,022, +0,058] | −0,014 [−0,030, +0,007] | +0,036 [−0,012, +0,073] | [−0,055, +0,107] |
| RF | m1 | −0,005 | −0,024 | +0,020 [−0,021, +0,056] | [−0,073, +0,092] |
| MLP | m1 | +0,006 | −0,005 | +0,011 [−0,032, +0,056] | [−0,062, +0,086] |
| GB | m4 ROC-AUC | −0,054 [−0,071, −0,032] | −0,000 | −0,053 [−0,070, −0,032] | [−0,115, −0,001] |
| RF | m4 | −0,052 | −0,001 | −0,050 [−0,068, −0,030] | [−0,110, +0,001] |
| MLP | m4 | −0,060 | −0,003 | −0,058 [−0,075, −0,036] | [−0,127, −0,002] |

Kepekaan dengan 2023 di kelompok tenang (kab 200, dua tahap 500): GB m1 +0,019 [−0,017, +0,050], dua
tahap [−0,067, +0,092]; m4 −0,045. Per tahun GB m1, per kab: 2015 +0,102, 2014 −0,046, 2019 +0,071;
tenang +0,034 +0,024 +0,115 −0,059 −0,015 (2023 +0,088). Nasional (411 kab): +0,048 +0,009 +0,033;
+0,070 +0,046 +0,081 −0,020 +0,039 (2023 +0,013).

**R9.** Selang dua tahap m1 pada pasangan Tabel 2 mencakup nol (+0,036 [−0,055, +0,107]); m4 berlawanan
tanda (−0,053). Naskah menulis: pada rujukan benchmark, pola tahun ekstrem tidak dapat dibedakan dari nol,
dan tandanya bergantung pada rujukan (MC +0,293, EF-3m −0,010, EF-1m +0,070, pasangan +0,036) dan ukuran.

### C7. Platt pada tahun sesudah 2015

| Tahun | Prevalensi | ECE mentah | ECE Platt semua | ECE Platt pasca-2015 | rata-rata p mentah / pasca-2015 |
|---|---|---|---|---|---|
| 2020 | 3,49 % | 0,040 | 0,052 | 0,019 | 7,52 % / 5,40 % |
| 2021 | 1,68 % | 0,019 | 0,031 | 0,012 | 3,63 % / 2,89 % |
| 2022 | 1,07 % | 0,007 | 0,014 | 0,005 | 1,68 % / 1,32 % |
| 2023 | 11,58 % | 0,027 | 0,021 | 0,046 | 8,84 % / 7,01 % |
| 2024 | 3,75 % | 0,021 | 0,032 | 0,012 | 5,89 % / 4,90 % |
| 2025 | 2,55 % | 0,024 | 0,035 | 0,018 | 4,92 % / 4,38 % |
| Rata-rata biasa | 4,0 % | 0,023 | 0,031 | 0,019 | |
| 2015 / 2014 / 2019 ditahan | 22,6 / 17,7 / 15,7 % | 0,058 / 0,102 / 0,021 | 0,033 / 0,083 / 0,011 | 0,052 / 0,097 / 0,032 | |
| Rata-rata ekstrem | | 0,060 | 0,042 | 0,060 | |

Tahun pasca-2015 dalam lipatan 2020 hanya empat (2016–2019). |rata-rata p − prevalensi| mengecil pada 5
dari 6 tahun biasa (tidak pada 2023, tahun El Niño dengan prevalensi 11,6 %).

**R14 — lolos.** ECE tahun biasa 0,019 < 0,023 dan 5 dari 6 tahun. Naskah menulis bahwa prakiraan
berlebih di tahun tenang konsisten dengan pergeseran rezim sesudah 2015: pemeta dari tahun pasca-2015
mengurangi ECE tahun biasa, tetapi memperburuk 2023 dan tidak menolong tahun ekstrem (0,060), jadi tidak
ada satu pemeta dari tahun lampau yang benar untuk kedua jenis tahun.

### C4. Siaga dan nilai keputusan atas EF-1m (per kabupaten, benih 42)

Kuota bulanan (10 % kabupaten tertinggi tiap bulan), hit rate [kab 500] [dua tahap 1.000] / FAR:

| Model | Biasa: hit | FAR | Ekstrem: hit | FAR |
|---|---|---|---|---|
| GB | 0,405 [0,360, 0,448] [0,304, 0,513] | 0,868 | 0,244 [0,229, 0,260] [0,203, 0,285] | 0,551 |
| RF | 0,435 [0,395, 0,478] [0,324, 0,545] | 0,858 | 0,241 | 0,558 |
| MLP | 0,413 | 0,868 | 0,239 | 0,558 |
| Local history | 0,314 | 0,892 | 0,213 | 0,607 |
| **EF-1m** | 0,350 [0,307, 0,399] [0,263, 0,446] | 0,877 | 0,228 | 0,578 |
| EF-3m | 0,282 | 0,894 | 0,219 | 0,593 |
| MC | 0,184 | 0,928 | 0,202 | 0,626 |

Selisih hit rate atas EF-1m [kab] [dua tahap]: biasa GB +0,055 [+0,005, +0,112] [−0,043, +0,184], RF
+0,085 [+0,030, +0,138] [−0,014, +0,217], MLP +0,063 [+0,012, +0,110] [−0,038, +0,193], local history
−0,036; ekstrem GB +0,016 [+0,001, +0,028] [−0,004, +0,033], RF +0,013, MLP +0,011. Per tahun GB / EF-1m:
2020 0,477/0,372, 2021 0,458/0,337, 2022 0,509/0,264, 2023 0,229/0,207, 2024 0,297/0,395,
2025 0,460/0,524; 2015 0,201/0,198, 2014 0,282/0,262, 2019 0,248/0,224.

Ambang 10 % luar-lipatan (aturan a), hit / FAR / bagian siaga: biasa GB 0,365 / 0,658 / 5,1 %, EF-1m
0,407 / 0,790 / 8,9 %, EF-3m 0,434 / 0,808 / 10,7 %; ekstrem GB 0,732 / 0,379 / 22,1 %, EF-1m 0,357 /
0,212 / 8,3 %.

Nilai ekonomi relatif, a = 0,05 / 0,1 / 0,2 / 0,3 / 0,5:

| | Biasa | Ekstrem |
|---|---|---|
| GB, siaga bila p > a | +0,380 / +0,219 / +0,077 / −0,001 / −0,021 | +0,285 / +0,468 / +0,579 / +0,487 / +0,293 |
| EF-1m, p > a | +0,031 / −0,668 / −1,364 / −1,692 / −1,655 | +0,119 / +0,245 / +0,440 / +0,314 / +0,080 |
| EF-3m, p > a | −0,122 / −0,741 / −1,390 / −1,225 / −0,182 | +0,312 / +0,417 / +0,399 / +0,183 / 0,000 |
| GB, ambang luar-lipatan | +0,283 / +0,224 / +0,130 / +0,081 / +0,009 | +0,367 / +0,461 / +0,568 / +0,442 / +0,287 |
| EF-1m, ambang luar-lipatan | +0,011 / −0,140 / −0,307 / −0,048 / 0,000 | −0,477 / −0,051 / +0,042 / +0,028 / 0,000 |
| GB, amplop | +0,457 / +0,344 / +0,194 / +0,127 / +0,069 | +0,459 / +0,555 / +0,638 / +0,558 / +0,391 |
| EF-1m, amplop | +0,358 / +0,267 / +0,181 / +0,133 / +0,064 | +0,317 / +0,492 / +0,605 / +0,511 / +0,335 |

Selisih GB − EF-1m, p > a [dua tahap] (selang kabupaten semua > 0): biasa +0,348 [+0,068, +0,784],
+0,887 [+0,159, +2,028], +1,441 [+0,355, +3,174], +1,691 [+0,416, +3,694], +1,634 [+0,443, +3,346];
ekstrem +0,166 [−0,104, +0,475], +0,222 [−0,001, +0,489], +0,139 [−0,032, +0,292], +0,173 [−0,033,
+0,337], +0,213 [−0,029, +0,375]. Ambang luar-lipatan, dua tahap: biasa [−0,018, +0,716], [+0,019, +0,861],
[+0,095, +0,965], [−0,004, +0,292], [−0,014, +0,038]; ekstrem semua > 0.

Nasional, kuota bulanan (498 kab): biasa GB 0,646 / FAR 0,638, RF 0,656, EF-3m 0,622, EF-1m 0,621, MC
0,521; GB − EF-3m +0,024 [+0,005, +0,046] [−0,007, +0,060], RF − EF-3m +0,034 [+0,014, +0,053]
[+0,004, +0,067]. Ekstrem GB 0,396, EF-3m 0,386; GB − EF-3m +0,010 [−0,002, +0,021 dua tahap].

**R11.** Kuota per kabupaten: selang dua tahap selisih hit GB − EF-1m mencakup nol di kedua kelompok →
naskah memberi angka (41 % lawan 35 % di tahun biasa, 24 % lawan 23 % di tahun ekstrem) **tanpa** klaim
"menangkap lebih banyak daripada kalender kabupaten". Angka lama "28 %" adalah EF-3m. Nilai ekonomi,
siaga bila p > a: lolos pada kelima a di tahun biasa, tidak pada satu pun a di tahun ekstrem. Bacaan:
keunggulan di tahun biasa datang dari kegagalan EF-1m sebagai peluang (frekuensi dari tahun latih yang
memuat tahun ekstrem menyiagakan terlalu banyak di tahun tenang: nilai sampai −1,69), bukan dari
diskriminasi — pada amplop GB − EF-1m hanya +0,099 / +0,077 / +0,013 / −0,006 / +0,005.

### C3. Rujukan dipilih di dalam lipatan

AUC-PR dalam-tahun luar-lipatan pada tahun latih (MC / EF-3m / EF-1m), pilihan per lipatan:

- Per kabupaten: EF-1m di kesembilan lipatan (S1 2020: 0,351 / 0,435 / 0,456; … S1 2025: 0,262 / 0,352 /
  0,389; S2 2015, 2014, 2019: EF-1m 0,341, 0,350, 0,343 lawan EF-3m 0,295, 0,307, 0,296).
- Nasional: EF-3m di kesembilan lipatan, dengan jarak kecil atas EF-1m (S1 2020: 0,764 lawan 0,752; S1
  2025: 0,744 lawan 0,742; S2 2015: 0,732 lawan 0,732).

Karena pilihannya sama dengan pasangan Tabel 2 di setiap lipatan, Tabel 2 terhadap rujukan per lipatan
identik dengan C1 (per kabupaten GB +0,031 [−0,002, +0,066], dua tahap [−0,031, +0,101]; nasional RF
+0,050, dua tahap [+0,019, +0,084]); S2 per tahun sama dengan Hasil_Patokan_7.

**R10.** Sama di keenam lipatan S1 (dan ketiga S2) untuk kedua sasaran → Tabel 2 tetap. Naskah menulis
bahwa aturan "calon klimatologi dengan AUC-PR dalam-tahun luar-lipatan tertinggi pada tahun latih"
memilih rujukan yang sama di setiap lipatan, dan aturan itu menjadi rujukan tetap benchmark. Pilihan
pasca-hoc dari tahun uji (Hasil_Patokan_6) dengan demikian tidak mengubah angka.

### C6. Baseline sederhana yang lebih kuat (benih 42)

| | Per kab: AP | atas EF-1m [kab] [dua tahap] | Nasional: AP | atas EF-3m [kab] [dua tahap] |
|---|---|---|---|---|
| `log_ef1_jeda` | 0,246 | +0,002 [−0,027, +0,031] [−0,055, +0,051] | 0,713 | +0,025 [+0,009, +0,042] [−0,011, +0,067] |
| `xgb_sejarah` | 0,219 | −0,025 [−0,051, −0,003] [−0,085, +0,027] | 0,699 | +0,011 [−0,008, +0,030] [−0,023, +0,050] |
| `log_klim_jeda` | 0,201 | −0,044 [−0,070, −0,021] [−0,096, −0,001] | 0,674 | −0,014 [−0,035, +0,007] [−0,051, +0,022] |

Per tahun atas rujukan, `log_ef1_jeda`: per kab +0,000 +0,030 +0,022 +0,037 −0,102 +0,022; nasional
+0,098 +0,015 +0,035 +0,027 −0,042 +0,015. `xgb_sejarah`: per kab −0,040 −0,005 −0,010 +0,071 −0,105 −0,062;
nasional +0,054 +0,012 +0,050 +0,002 −0,043 −0,010.

Selisih model penuh − baseline baru [kab] [dua tahap]:

| | Per kab | Nasional |
|---|---|---|
| GB − `log_ef1_jeda` | +0,030 [+0,011, +0,052] [−0,016, +0,076] | +0,014 [+0,002, +0,024] [−0,013, +0,037] |
| GB − `xgb_sejarah` | +0,057 [+0,035, +0,088] [+0,022, +0,114] | +0,027 [+0,019, +0,035] [+0,013, +0,043] |
| RF − `log_ef1_jeda` | +0,041 [+0,024, +0,061] [+0,001, +0,082] | +0,025 [+0,015, +0,036] [−0,000, +0,048] |
| RF − `xgb_sejarah` | +0,068 [+0,050, +0,093] [+0,038, +0,113] | +0,039 [+0,029, +0,048] [+0,025, +0,053] |

S2, atas rujukan (`log_ef1_jeda` / `xgb_sejarah` / GB): per kab 2015 +0,053 / +0,060 / +0,102, 2014
−0,010 / −0,011 / −0,046, 2019 +0,023 / −0,011 / +0,071; nasional 2015 +0,034 / +0,049 / +0,048, 2014
−0,008 / −0,001 / +0,009, 2019 −0,012 / +0,012 / +0,033.

**R13.** Selang dua tahap GB − `log_ef1_jeda` mencakup nol untuk **kedua** sasaran → klaim model
dipersempit: gradient boosting tidak dapat dibedakan dari regresi logistik pada kalender kabupaten (EF-1m)
ditambah apakah bulan lalu melampaui ambang dan hitungan bulan lalu. `log_ef1_jeda` masuk Tabel 2 sebagai
pembanding utama kedua. Random forest tetap di atasnya per kabupaten (batas bawah +0,001) dan menyentuh nol
nasional. Model pohon pada sejarah api dan musim saja (`xgb_sejarah`) di bawah model penuh dengan selang
dua tahap di atas nol, jadi fitur lain menambah sesuatu pada pohon, tetapi tidak lebih daripada yang sudah
diberikan baseline logistik sederhana itu. Di 2014 ditahan, kedua baseline baru di atas GB per kabupaten.

### C5. Kebocoran ambang terhadap rujukan sepadan (benih 42)

Label ulang dihitung sama persis dengan berkas kepekaan lama (kecocokan y 1,0000). Keunggulan atas
rujukan pasangan Tabel 2, S1 2020–2025 [kab 1.000] [dua tahap 1.000]:

| | Label baku | Label ulang, kabupaten tetap | Label ulang, kabupaten per lipatan |
|---|---|---|---|
| Per kab, prevalensi / AP EF-1m | 4,02 % / 0,245 (411 kab) | 3,73 % / 0,247 | 3,75 % / 0,247 (407–409 kab) |
| RF | +0,043 [+0,016, +0,071] [−0,013, +0,098] | +0,031 [−0,001, +0,063] [−0,020, +0,083] | +0,031 [−0,003, +0,065] [−0,021, +0,082] |
| GB | +0,031 [−0,002, +0,066] [−0,031, +0,101] | +0,028 [−0,014, +0,069] [−0,041, +0,113] | +0,028 [−0,016, +0,071] [−0,040, +0,118] |
| MLP | +0,012 [−0,021, +0,042] [−0,044, +0,076] | −0,007 [−0,042, +0,026] [−0,056, +0,052] | −0,007 [−0,044, +0,027] [−0,060, +0,054] |
| S2 GB 2015 / 2014 / 2019 | +0,102 / −0,046 / +0,071 | +0,074 / −0,062 / +0,047 | +0,070 / −0,062 / +0,046 |
| Nasional, prevalensi / AP EF-3m | 6,13 % / 0,688 | 5,39 % / 0,694 | — |
| RF | +0,050 [+0,032, +0,067] [+0,019, +0,084] | +0,053 [+0,033, +0,073] [+0,013, +0,099] | |
| GB | +0,038 [+0,021, +0,054] [+0,005, +0,074] | +0,043 [+0,024, +0,061] [+0,005, +0,083] | |
| MLP | +0,026 [+0,007, +0,044] [−0,006, +0,058] | +0,031 [+0,010, +0,051] [−0,005, +0,070] | |
| S2 GB 2015 / 2014 / 2019 | +0,048 / +0,009 / +0,033 | +0,046 / +0,004 / +0,031 | |

**R12 — terpicu untuk per kabupaten.** MLP berubah dari +0,012 ke −0,007 (lebih dari setengah, tanda
berbalik); RF turun 0,012 dan selang kabupatennya kini menyentuh nol. Tidak ada batas bawah dua tahap
yang berbalik tanda (semua per kabupaten sudah di bawah nol, semua nasional RF/GB tetap di atas nol).
Masuk badan naskah (6.1) sebagai batasan: dengan ambang dari tahun latih, keunggulan per kabupaten
lebih kecil lagi (RF +0,031, GB +0,028) dan tidak satu pun selang kabupatennya di atas nol; nasional
tidak berubah berarti.

## Putusan fase C dan akibat untuk naskah

1. **Per kabupaten, model yang dipelajari tidak terbukti lebih baik daripada kalender kabupaten (EF-1m)**
   bila tahun diambil ulang (R8), juga dengan ambang dari tahun latih (R12), dan gradient boosting tidak
   dapat dibedakan dari logistik sederhana EF-1m + bulan lalu (R13). Kuota bulanan menangkap 41 % lawan
   35 % (R11, tanpa klaim). Klaim yang tersisa per kabupaten: tidak ada keunggulan yang dapat dibela.
2. **Nasional, random forest dan gradient boosting tetap di atas EF-3m** dengan selang dua tahap di atas
   nol (+0,050, +0,038), tahan terhadap ambang dari tahun latih; tetapi gradient boosting tidak dapat
   dibedakan dari `log_ef1_jeda` (+0,014 [−0,013, +0,037]) dan RF menyentuh nol (+0,025 [−0,000, +0,048]).
3. **Rujukan:** aturan per lipatan memilih EF-1m dan EF-3m di setiap lipatan (R10); pilihan tidak lagi
   pasca-hoc dalam akibatnya.
4. **Tahun ekstrem:** tidak dapat dibedakan dari nol pada rujukan benchmark (R9).
5. **Nilai keputusan:** siaga bila p > a, GB jauh lebih bernilai daripada EF-1m di tahun biasa karena
   frekuensi lampau terlalu tinggi di tahun tenang, bukan karena diskriminasi (amplop hampir sama).
6. **Kalibrasi:** pemeta dari tahun sesudah 2015 memperbaiki tahun biasa, tidak tahun ekstrem (R14).

Akibat: tesis naskah berubah dari "ML menambah keterampilan sedang" menjadi **"pada benchmark terbuka
dengan klimatologi sepadan, ML hampir tidak menambah apa pun per kabupaten; yang tersisa (nasional, kecil)
dapat dicapai dengan regresi logistik pada kalender kabupaten dan bulan lalu"**. Ini hasil negatif yang
jelas dan dapat dibela; sumbangan utama tetap benchmark (data, rujukan per lipatan, protokol). Judul,
abstrak, pendahuluan, 6.1, 6.2, 6.7, diskusi, dan kesimpulan ditulis ulang menurut aturan R8–R14.

Tambahan C7 (dihitung dari tabel di atas, untuk Tabel 7 naskah): rata-rata peluang Platt pasca-2015,
tahun biasa (5,40 + 2,89 + 1,32 + 7,01 + 4,90 + 4,38) / 6 = 4,32 %; tahun ekstrem (17,44 + 7,99 + 12,49)
/ 3 = 12,64 %. Keunggulan atas MC per model (= keunggulan atas rujukan + selisih rujukan, C1): per
kabupaten RF +0,162, GB +0,150, MLP +0,131; nasional RF +0,259, GB +0,247, MLP +0,235.
