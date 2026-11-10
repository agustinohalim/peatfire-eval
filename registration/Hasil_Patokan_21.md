# Hasil Patokan 21 — protokol evaluasi musim 2026 (didaftarkan sebelum data)

Didaftarkan 9 November 2026, sebelum data VIIRS S-NPP pemrosesan standar (`VIIRS_SNPP_SP`) untuk 2026 tersedia dan
sebelum satu pun angka 2026 dihitung. Keputusan penulis 9 Nov sesudah telaah internal pra-kirim kedua puluh
(catatan telaah tidak dirilis): empat rujukan, gabung 2020–2026. Berkas ini menggantikan aturan 2026 yang asimetris di naskah (EF-LD wajib nasional,
per kabupaten hanya dilaporkan) dan menambah aturan gabung yang sebelumnya tidak terdaftar. Tidak ada analisis
yang dijalankan bersama pendaftaran ini; tidak ada angka 2020–2025 yang berubah.

## Data dan pembagian

- Deteksi: `VIIRS_SNPP_SP` 2026 setelah dirilis, kotak, filter dan poligon kabupaten sama dengan panel utama
  (Bagian 3–4.4 naskah). Data hampir-waktu-nyata tidak dipakai.
- Skema S1: setiap tahun uji dilatih pada tahun sebelumnya saja; 2026 dilatih pada 2012–2025. Ambang, kabupaten yang
  dinilai, kalender, EF-P, EF-L dan EF-LD dihitung dari tahun latih fold itu, seperti fase M (Hasil_Patokan_18).
- Model: dua puluh dua model naskah ditambah GB + hitungan nasional, dengan hiperparameter dan benih 42 seperti
  terarsip. Tidak ada model, fitur atau rujukan yang ditambahkan sesudah data 2026 terlihat.
- Kode: rilis peatfire-eval yang dirujuk naskah final, tanpa perubahan.

## Rujukan aturan

Untuk kedua sasaran (per kabupaten dan nasional) sama: kalender mentah (EF-1m per kabupaten, EF-3m nasional), EF-P,
EF-L, EF-LD. Alasan: EF-LD mengalahkan EF-L nasional pada VIIRS (fase M) dan per kabupaten pada prakiraan maju MODIS
(fase N); satu aturan untuk kedua sasaran menghindari memilih rujukan menurut sasaran tempat rujukan itu menang.

## Aturan keputusan (R48)

- Tahun uji: 2020–2026 (tujuh tahun), masing-masing dinilai seperti S1.
- Ukuran: Brier skill score model atas tiap rujukan dalam setiap tahun; agregasi utama = rata-rata rasio tahunan.
  Satu rasio atas semua baris dan median tahun dilaporkan di sampingnya, deskriptif.
- Selang: dua tahap (tahun lalu kabupaten, 1.000 ulangan, benih 2026) dan t95 atas tujuh tahun.
- **R48(a)** Suatu model dilaporkan mengalahkan rujukan untuk suatu sasaran hanya bila, atas keempat rujukan, batas
  bawah dua tahap dan batas bawah t95 keduanya > 0, dan batas bawah dua tahap > 0 di kelima benih {2026, 1, 2, 3, 4}
  (lolos kokoh). Lolos tanpa kokoh dilaporkan sebagai lolos rapuh, bukan keunggulan.
- **R48(b)** Bila tidak ada model yang lolos kokoh untuk suatu sasaran, laporan menulis bahwa tidak ada model yang
  terbukti mengalahkan keempat rujukan pada 2020–2026 untuk sasaran itu, dengan daya deteksi (MDD) tujuh tahun.
- **R48(c)** 2026 sendiri: BSS tiap model atas tiap rujukan dengan selang bootstrap kabupaten saja (tanpa selang
  tahun), deskriptif; tidak dapat menetapkan atau membatalkan keunggulan.
- **R48(d)** Rujukan: BSS EF-L atas kalender dan EF-LD atas EF-L pada 2026 dan pada 2020–2026, dengan selang yang
  sama. Bila EF-L di bawah kalender pada 2026 (BSS < 0), laporan menulis bahwa penyesuaian tingkat tidak membantu pada
  musim itu.
- **R48(e)** Bagian tingkat (offset periode dan tingkat tiap tahun) untuk GB atas kalender mentah dihitung untuk 2026
  dan 2020–2026, seperti fase I/N.
- Yang tidak berubah oleh 2026: semua angka dan vonis 2020–2025 dan MODIS di naskah ini.

## Tambahan bertanggal (9 Nov 2026, sesudah telaah internal kedua puluh satu, masih sebelum data standar 2026)

Telaah internal kedua puluh satu meminta empat hal; semuanya dicatat di sini sebelum data
`VIIRS_SNPP_SP` 2026 terbit, dan tidak mengubah R48.

**Status data saat pendaftaran.** Data hampir-waktu-nyata VIIRS 2026, portal SiPongi+ dan status ENSO 2026 sudah publik
pada 9 Nov. Penulis tidak mengunduh dan tidak melihat hitungan deteksi 2026 (9 Nov); tidak satu pun dipakai
untuk menyusun aturan.

**Ambang.** Seperti tertulis di atas, ambang, kabupaten yang dinilai dan rujukan diambil dari tahun latih tiap fold
(bukan dari seluruh panel seperti S1 di naskah), sehingga angka 2020–2025 di bawah R48 akan sedikit berbeda dari angka
naskah. Kalimat naskah "scored as in S1" diganti.

**Model tambahan.** Daftar model 2026 juga memuat GB + hitungan nasional + DC (`xgb_nasdc`, fase O), dengan masukan dan
setelan seperti fase O. Tidak ada model lain.

**Daya dan keunggulan 2026 yang dibutuhkan (proyeksi, bukan analisis data baru).** Dari rata-rata dan simpangan baku
enam tahun BSS atas EF-L (Hasil_Patokan_12, tabel R33; ambang seluruh panel), dengan anggapan sebaran per tahun sama:
- MDD80 tujuh tahun ≈ 0,889 × MDD80 enam tahun (rasio (t0,975 + t0,80)/√n untuk n = 7 lawan 6), yaitu ≈ 0,10–0,21
  (RF 0,120 / 0,097; GB 0,138 / 0,213; GB tersetel 0,180 / 0,148; GB + nasional 0,136 / 0,140; per kab / nasional).
- Nilai BSS 2026 atas EF-L yang membuat batas bawah t95 tujuh tahun > 0: tidak ada untuk RF, GB, GB tersetel (kedua
  sasaran) dan GB + nasional nasional — satu tahun ekstrem memperbesar sebaran lebih cepat daripada menaikkan rata-rata;
  hanya GB + nasional per kabupaten, bila keunggulan 2026 antara +0,092 dan +0,345. Jadi di bawah R48 uji 2026 hampir
  hanya dapat membantah, bukan mengukuhkan, keunggulan atas EF-L.

**Ramalan tanda 2026 saja (R48f, dapat dibantah oleh satu musim).** (i) BSS EF-L atas kalender mentah pada 2026 > 0
untuk kedua sasaran. (ii) Per kabupaten, bagian keunggulan GB atas kalender mentah yang hilang oleh kedua geser tingkat
pada 2026 ≥ 0,5. Bila (i) atau (ii) gagal, laporan menulis bahwa temuan tingkat tidak berulang pada 2026.

## Tambahan bertanggal kedua (9 Nov 2026, sesudah telaah internal kedua puluh dua, masih sebelum data standar 2026)

Telaah internal kedua puluh dua menilai kedua ramalan tanda R48(f) terutama menguji
apakah 2026 musim sepi, padahal status ENSO 2026 sudah publik; dan ramalan (ii) tak terdefinisi bila penyebutnya
mendekati nol. R48(a)–(e) tidak berubah; R48(f) diganti R48(f′).

**Status ENSO/IOD 2026, dicatat saat menulis tambahan ini.** Diunduh 9 Nov 2026 pukul 16.32 WIB dari NOAA CPC
(`oni.ascii.txt`) dan NOAA PSL (`dmi.had.long.data`, HadISST), dengan klasifikasi sama dengan fase M (Hasil_Patokan_18,
M3): ONI AMJ 2026 +0,95 (El Niño), MJJ +1,39, JJA +1,80, JAS +2,16, ASO +2,46; DMI Mei 2026 +0,146 (Juni belum ada di
berkas; rata-rata Mei–Juni tidak dapat dihitung). Jadi musim 2026 adalah El Niño kuat. Penulis kini mengetahui ini;
hitungan deteksi 2026 tetap tidak diunduh dan tidak dilihat.

**Musim sepi atau ramai, ditetapkan dari data 2026 sendiri, bukan dari ENSO.** Untuk tiap sasaran, prevalensi 2026 =
bagian baris (kabupaten-bulan per kabupaten; bulan nasional) yang berisi peristiwa, dengan ambang dan kabupaten yang
dinilai dari fold 2026 (tahun latih 2012–2025). 2026 **sepi** untuk sasaran itu bila prevalensinya di bawah median
prevalensi tahunan 2012–2025 dengan ambang yang sama; selain itu **ramai**. Median dihitung bersama analisis 2026,
dengan skrip yang sama, bukan sekarang.

**R48(f′) Ramalan tanda 2026 saja, bersyarat.**
- (i) Bila 2026 sepi untuk suatu sasaran: BSS EF-L atas kalender mentah pada 2026 > 0 untuk sasaran itu. Bila ramai:
  tidak ada ramalan tanda untuk EF-L atas kalender. Alasannya dari tabel M3 (Hasil_Patokan_18): per kabupaten
  EF-L di bawah kalender pada 5 dari 6 tahun dengan prevalensi ≥ 8,5 % (2009, 2012, 2014, 2019, 2023), tetapi di atasnya
  pada 2015 (ONI ASO +2,02), sedangkan 2023 (ASO +1,50) di bawahnya; nasional tandanya campur pada tahun ramai. BSS EF-L atas kalender 2026
  tetap dilaporkan untuk kedua cabang (R48d).
- (ii) Per kabupaten, bagian keunggulan GB atas kalender mentah pada 2026 yang hilang oleh kedua geser tingkat ≥ 0,5,
  dinilai hanya bila BSS GB atas kalender mentah per kabupaten pada 2026 > 0 dengan batas bawah selang bootstrap
  kabupaten > 0. Bila tidak, ditulis bahwa bagian tingkat 2026 tidak terdefinisi, bukan bahwa ramalan gagal. Berlaku
  untuk musim sepi maupun ramai.
- Bila ramalan yang dinilai gagal, laporan menulis bahwa temuan tingkat tidak berulang pada 2026 untuk kondisi itu.

**Musim pertama yang sepenuhnya ex-ante.** 2026 didaftarkan sesudah musim kebakarannya berlangsung dan sesudah status
ENSO-nya publik, jadi bukan uji buta. Untuk 2027, aturan yang sama (R48(a)–(e) dengan 2020–2027 dan R48(f′))
didaftarkan dengan stempel waktu publik sebelum 1 Juni 2027, sebelum musim kemarau, sehingga 2027 adalah musim
pertama yang sepenuhnya ex-ante.

**Stempel waktu publik.** Berkas ini, kedua tambahannya dan kode fase O masuk rilis peatfire-eval 2.11.0 di Zenodo
sebelum data `VIIRS_SNPP_SP` 2026 terbit, sehingga waktu pendaftaran dapat diperiksa di luar repositori penulis.

## Tambahan bertanggal ketiga (9 Nov 2026, sesudah panel 23, masih sebelum data standar 2026)

Pemicu: telaah internal pra-kirim kedua puluh tiga (catatan telaah tidak dirilis): di Riau dan Sumatra bagian utara puncak api jatuh pada
Februari–Maret, dan sasaran per kabupaten menilai kedua belas bulan, jadi pendaftaran sebelum 1 Juni 2027 tidak
mendahului musim api semua kabupaten. Perubahan satu-satunya: pendaftaran 2027 (aturan yang sama seperti tambahan
kedua) dilakukan dengan stempel waktu publik **sebelum 1 Januari 2027**, bukan sebelum 1 Juni 2027. Ini hanya
memajukan tanggal, jadi lebih ketat; aturan 2026 (R48(a)–(e), R48(f′)) tidak berubah. Penulis tetap belum melihat
hitungan deteksi 2026.

## Tambahan bertanggal keempat (10 Nov 2026, sesudah panel 25, masih sebelum data standar 2026)

Pemicu: telaah internal pra-kirim kedua puluh lima (catatan telaah tidak dirilis): ramalan (ii) R48(f′) menuntut bagian
tingkat per kabupaten ≥ 0,5, padahal menurut relasi fase Q (Hasil_Patokan_23) dan R (Hasil_Patokan_24) tingkat yang
diambil hampir nol pada tahun yang lebih ramai dari tahun latih; untuk 2026 (El Niño kuat) (ii) jadi diharapkan gagal
atau tak terdefinisi, sehingga tidak membedakan apa pun. R48(a)–(e) dan R48(f′) (i), (ii) tidak berubah dan tetap
dilaporkan. Ditambah satu ramalan yang menguji relasi itu sendiri. Penulis tetap belum melihat hitungan deteksi 2026.

**R48(f′)(iii) Ramalan relasi tingkat 2026, per kabupaten.** Dengan definisi persis R1 (Hasil_Patokan_24) pada fold
2026 (ambang, kabupaten dan kalender dari tahun latih 2012–2025; GB benih 42): L₂₀₂₆ = BSS GB atas kalender mentah −
BSS sesudah kedua prakiraan digeser ke laju teramati 2026; e₂₀₂₆ = logit(rata-rata kalender 2026) − logit(laju
teramati 2026).
- Bila e₂₀₂₆ ≤ 0 (kalender terlalu rendah): ramalan |L₂₀₂₆| ≤ 0,05. Dasarnya: keenam tahun dengan e < 0 (VIIRS 2023,
  MODIS 2009, 2012, 2014, 2015, 2019) memberi L −0,036 sampai −0,007.
- Bila e₂₀₂₆ > 0: ramalan L₂₀₂₆ di dalam selang prakiraan 95 % regresi L atas e pada tahun dengan e > 0 saja (VIIRS
  lima tahun, MODIS maju tujuh tahun), intersep per rancangan, intersep VIIRS dipakai untuk 2026; selang prakiraan
  OLS biasa dengan derajat bebas n − 3. Koefisien dihitung dari log R1 yang sudah ada
  (`patokan_tingkat_r1_kab.log`, `patokan_modis_r_kab.log`) dengan `patokan_ramalan_2026.py`, dicatat di bawah
  sebelum data 2026 diunduh.
- Bila ramalan gagal, laporan menulis bahwa relasi tingkat tidak berulang pada 2026. Sasaran nasional tidak diramal
  (bagiannya tak presisi, 0,57 [0,06, 0,77]).

Koefisien (f0b9159 mendaftarkan aturan; skrip dijalankan 10 Nov 2026 sesudahnya, sebelum data 2026; log
`DL_FIRE_NASIONAL/penuh/patokan_ramalan_2026.log`). Tahun dengan e > 0: n = 12 (VIIRS 5, MODIS 7), df 9.
L̂ = +0,0342 + 0,2607 e (intersep VIIRS; MODIS −0,0399), kemiringan [+0,182, +0,339]; simpangan baku sisa s = 0,0559;
t(0,975; 9) = 2,2622. Selang prakiraan: L̂ ± 2,2622 · s · √(1 + x₀ᵀ(XᵀX)⁻¹x₀), x₀ = (e, 1, 0), dengan
(XᵀX)⁻¹ untuk (e, VIIRS, MODIS) = [[0,388023, −0,582034, −0,334753], [−0,582034, 1,073051, 0,502129],
[−0,334753, 0,502129, 0,431653]]. Contoh: e = 0,5: +0,165 [+0,005, +0,324]; e = 1,0: +0,295 [+0,151, +0,439];
e = 1,5: +0,425 [+0,287, +0,564]; e = 2,0: +0,556 [+0,412, +0,699]. Tahun dengan e ≤ 0: L −0,036 sampai −0,007, maks
|L| 0,036, jadi batas 0,05.
