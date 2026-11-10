# Hasil Patokan 24 — fase R revisi (galat tingkat model sendiri; klaster provinsi untuk Tabel 4)

**10 November 2026.** Lanjutan `Hasil_Patokan_23.md`, untuk `Naskah_Artikel2_Revisi.md`.
Pemicu: telaah internal pra-kirim kedua puluh empat atas draf revisi (catatan telaah tidak dirilis).
(1) Metode X1, advokat setan X4, domain D5, editor X2: hubungan Q1 (L atas galat tingkat kalender) sebagian besar
tersirat dari definisinya; yang belum dilaporkan adalah galat tingkat model sendiri. (2) Metode X3/X5, penyajian X3:
bootstrap klaster provinsi fase Q hanya mencakup sebagian vonis; sel Tabel 4 (ambang seluruh panel, termasuk EF-D,
EF-LD dan GB + nasional ± DC) belum diuji.

## Rancangan

**R1 — galat tingkat model.** Data dan prakiraan persis Q1 (VIIRS S1 ambang tahun latih, GB benih 42; MODIS maju,
GB-FH enam benih). Per tahun: e_k = logit(rata-rata kalender) − logit(laju teramati), e_m = logit(rata-rata
prakiraan GB) − logit(laju teramati) (MODIS: rata-rata atas benih dari logit rata-rata tiap benih). Regresi gabungan
18 tahun per sasaran, intersep per rancangan, bootstrap tahun dalam rancangan 10.000 ulangan benih 2026:
(A) L atas |e_k| (ulang Q1); (B) L atas |e_k| − |e_m|. Dilaporkan: kemiringan, selang, R² masing-masing; e_m per tahun.

**R2 — klaster provinsi untuk Tabel 4.** Data persis `patokan_tingkat_o.py` (VIIRS S1 2020–2025, ambang seluruh panel,
semua kolom KOL). Untuk tiap model/rujukan di Tabel 4 atas kalender, EF-P, EF-L dan EF-LD (sel yang dinilai di Tabel 4):
batas dua tahap kabupaten dan provinsi (34 provinsi, 1.000 ulangan, benih 2026; 10.000 bila batas provinsi dalam
±0,01 dari nol) dan selang tahun t. Pemeriksaan: rata-rata dan batas kabupaten = Tabel 4.

Tidak diuji ulang (disebut di naskah): rancangan MODIS tinggalkan-satu-tahun, peringkat (AUC-PR), nilai peringatan.

Skrip: `patokan_tingkat_r.py` (R1 VIIRS dan R2; bagian data P dan O disalin tanpa perubahan isi) dan
`patokan_modis_r.py` (R1 MODIS dan regresi gabungan; bagian data O disalin tanpa perubahan isi).

Catatan sebelum menjalankan: R1 VIIRS ditulis di berkas sendiri, `patokan_tingkat_r1.py` (bagian data P tanpa
perubahan isi; menulis tabel yang dibaca `patokan_modis_r.py`), dan `patokan_tingkat_r.py` hanya memuat R2. Isi
rancangan tidak berubah.

## Aturan keputusan

- **R53 (R1).** Bila R² (B) > R² (A) dan kemiringan (B) > 0 dengan selang tanpa nol untuk kedua sasaran, §6.2 dan §9
  menulis bahwa bagian tingkat adalah galat tingkat kalender yang sudah dikoreksi model (selisih kedua galat), dengan
  angkanya; bila tidak, kalimat Q1 sekarang tetap dan e_m dilaporkan di suplemen. Tidak ada klaim baru di abstrak.
- **R54 (R2).** Sel Tabel 4 yang tebal (kedua selang tanpa nol) dan batas provinsinya ≤ 0 kehilangan cetak tebal dan
  diberi tanda; vonis teks yang bertumpu padanya ditulis tidak bertahan bila provinsi diambil ulang (seperti R51).
  Sel bertanda † atau ‡ dilaporkan, tanpa perubahan tanda kecuali batas provinsinya ≤ 0, yang ditambah catatan.
- **R55.** Bila abstrak berubah karena R53 atau R54, panel 25 dijalankan; bila tidak, tidak.

## Hasil

Skrip 84e9f5a; dijalankan 10 Nov 2026. Log: `DL_FIRE_NASIONAL/penuh/patokan_tingkat_r1_{kab,nas}.log`,
`DL_FIRE_MODIS/patokan_modis_r_{kab,nas}.log`, `DL_FIRE_NASIONAL/penuh/patokan_tingkat_r_{kab,nas}.log`.

**Pemeriksaan.** Regresi (A) mengulang Q1 persis: +0,2918 [+0,2365, +0,3414] per kab, +0,3347 [+0,1992, +0,4105]
nasional (Hasil_Patokan_23). Rata-rata dan batas kabupaten R2 sama dengan Tabel 4 untuk semua sel.

**R1.** e_m per tahun (logit rata-rata prakiraan GB − logit laju teramati):

| Rancangan | e_m per tahun |
|---|---|
| VIIRS per kab 2020–2025 | +0,723 +0,533 +0,274 −0,338 +0,491 +0,683 |
| VIIRS nasional | +0,796 +0,386 −0,059 −0,292 +0,263 +0,127 |
| MODIS maju per kab 2008–2019 | +0,048 +0,135 +1,132 +0,105 −0,259 +0,042 −0,156 −0,540 +0,740 +0,930 +0,065 −0,223 |
| MODIS maju nasional | +0,205 +0,001 +0,552 +0,009 +0,001 +0,068 −0,099 −0,266 +0,686 +0,480 +0,094 −0,124 |

GB juga terlalu tinggi di tahun sepi (e_m > 0), tetapi lebih kecil dari kalender (VIIRS per kab e_k +0,98 sampai
+2,12, e_m +0,27 sampai +0,72).

| Sasaran | (A) L atas abs(e_k) | R² | (B) L atas abs(e_k) − abs(e_m) | R² |
|---|---|---|---|---|
| per kab | +0,292 [+0,237, +0,341] | 0,863 | +0,363 [+0,274, +0,529] | 0,794 |
| nasional | +0,335 [+0,199, +0,411] | 0,725 | +0,464 [+0,379, +0,631] | 0,866 |

**R53: tidak terpenuhi.** Kemiringan (B) positif dengan selang tanpa nol di kedua sasaran, tetapi R² (B) < R² (A)
per kab. Kalimat Q1 tetap; e_m dan regresi (B) dilaporkan di suplemen (Tabel S16h a), dengan satu kalimat
deskriptif di §6.2 (tanpa klaim baru).

**R2.** 34 provinsi (per kab: 411 kabupaten bersasaran; nasional: 498). Batas bawah dua tahap (batas atas untuk sel
negatif), kabupaten / provinsi (10.000 ulangan dalam kurung):

| Sel | Tabel 4 | Kabupaten | Provinsi |
|---|---|---|---|
| per kab EF-LD atas EF-L (‡) | +0,018 | +0,001 | −0,008 (−0,009) |
| per kab riwayat lokal atas EF-L (‡, negatif; batas atas) | −0,147 | −0,005 | +0,002 |
| per kab GB + nasional atas EF-L (†) | +0,091 | +0,007 | −0,002 (−0,002) |
| per kab GB + nasional + DC atas EF-L (†) | +0,085 | +0,003 | −0,008 (−0,007) |
| per kab EF-P atas kalender (tidak tebal) | +0,076 | −0,001 | −0,0074 (−0,0055) |
| nasional EF-LD atas EF-L (tebal) | +0,034 | +0,012 | +0,0075 (+0,0071) |
| nasional RF atas EF-L (tidak tebal) | +0,058 | −0,000 | −0,011 |

Semua sel tebal lain (atas kalender, atas EF-P, EF-D di bawah EF-L) tetap dengan batas provinsi jauh dari nol
(per kab terkecil EF-D atas kalender +0,088; nasional terkecil EF-D atas kalender +0,052). Laju konstan (†, nasional)
tidak termasuk KOL fase O, jadi tidak diuji.

**R54.** Tak satu sel tebal kehilangan cetak tebal. Empat sel bertanda (per kab EF-LD atas EF-L ‡, riwayat lokal atas
EF-L ‡, GB + nasional ± DC atas EF-L †) kehilangan kelulusan dua tahap bila provinsi diambil ulang → tanda § dan
catatan di Tabel 4, kalimat §6.2 masing-masing, §6.3 (cakupan) dan §8. Untuk sel negatif (riwayat lokal) aturan
dibaca simetris (batas atas provinsi ≥ 0). Tak satu vonis teks bertumpu pada sel itu sebagai kemenangan: §6.2 sudah
menulis bahwa tak satu model lolos atas EF-L dan EF-LD per kab tak lolos kedua selang. EF-LD sebagai rujukan
nasional (+0,034) bertahan (batas provinsi +0,007).

**R55.** Abstrak tidak berubah → panel 25 tidak dijalankan.
