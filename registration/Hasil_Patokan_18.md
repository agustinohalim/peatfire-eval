# Hasil Tolok Ukur Nasional 18 — fase M revisi (rujukan kekeringan, galat Monte Carlo, blok tahun, fase ENSO)

**8 November 2026.** Lanjutan `Hasil_Patokan_17.md` (fase L), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim keempat belas atas draf revisi (catatan telaah tidak
dirilis). Butir yang perlu analisis: (4) tak satu pun dari tiga rujukan yang didaftarkan untuk 2026 memakai
kekeringan; (1) satu-satunya hasil pembanding positif (nasional, MODIS maju, batas bawah 0,008) belum diuji dengan
selang yang menghormati ketergantungan antar-lipatan maju; galat Monte Carlo bootstrap 1.000 ulangan tidak
dilaporkan padahal ada ujung selang dalam 0,01 dari nol (+0,0002, +0,008); (11) fase ENSO/IOD yang dapat
ditetapkan sebelum musim tidak dilaporkan untuk tahun uji maju.

Catatan yang sudah ada sebelum fase ini: kalender + cuaca bulan lalu sudah dinilai sebagai kolom peluang di
fase D/E (`log_ef1_fdrs` = logit EF-1m + DC + FFMC + musim; `ef1_cuaca` = logit EF-1m + DC, FFMC, hujan dan
anomali hujan bulan lalu). Atas EF-L per kabupaten keduanya negatif (−0,392 dan −0,202 rata-rata rasio,
`patokan_revisi_peluang_kab.log`). Keduanya bukan bentuk EF-L; yang diminta telaah adalah EF-L dengan
kekeringan kabupaten di samping atau sebagai ganti hitungan nasional. Itu belum pernah dinilai.

## Rancangan fase M (ditulis 8 Nov 2026, SEBELUM skrip fase M ditulis atau dijalankan)

Semua pasca-hoc; tahun uji VIIRS 2020–2025 dan tahun maju MODIS 2008–2019 sudah dinilai di fase sebelumnya.
Interval: dua tahap 1.000 ulangan, benih B.SEED, dan t atas tahun, kecuali disebut lain. "Lolos" = batas bawah
dua tahap dan batas bawah t keduanya > 0 (rata-rata rasio); satu rasio hanya punya selang dua tahap dan tidak
disebut "lolos" (R40c).

**M1 Rujukan kekeringan (VIIRS, S1 2020–2025, dua sasaran).** Skrip baru `patokan_tingkat_m.py`, fungsi rujukan
dan anomali nasional sama dengan `patokan_revisi_peluang.py`. Peubah kekeringan: DC bulan lalu kabupaten dari
`gfwed_komponen_MERRA2.CORRECTED.csv` (sama dengan fase D: geser satu bulan per kabupaten, hilang diisi median
kabupaten), diubah ke log1p(DC), anomali = log1p(DC bulan lalu) − rata-rata log1p(DC bulan lalu) kabupaten itu
pada bulan kalender yang sama di tahun latih (baris latih: tanpa tahunnya sendiri, seperti anomali nasional).
Dua bentuk tetap, regresi logistik C = 1e6 dipasang per lipatan di baris tahun latih seperti EF-L:
- EF-D: logit(ef1_halus) + anomali DC kabupaten (kekeringan menggantikan hitungan nasional);
- EF-LD: logit(ef1_halus) + anomali api nasional + anomali DC kabupaten (kekeringan di samping hitungan).
Dinilai: EF-D dan EF-LD atas EF-L, atas kalender (EF pasangan) dan atas EF-P; GB dan RF atas EF-LD; dan ulang
`log_ef1_fdrs`, `ef1_cuaca` atas EF-L. Untuk tiap sel tiga agregasi seperti fase L: (a) rata-rata rasio tahunan
dua tahap + t95, (b) satu rasio dua tahap, (c) median dan banyak tahun > 0. Koefisien tiap lipatan dicetak.

**M2 Galat Monte Carlo dan ketergantungan antar-tahun.**
(a) MODIS maju nasional (skrip baru `patokan_modis_m.py`, lipatan R sama persis dengan fase J/K, hanya
y_gabungan; galat kuadrat per tahun per kabupaten disimpan ke `DL_FIRE_MODIS/sse_modis_m_nas.npz`, tidak masuk
repo). GB atas EF-L, kalender, EF-P, rata-rata rasio dan satu rasio: dua tahap dengan lima benih
{B.SEED, 1, 2, 3, 4} × 1.000 ulangan, dan satu jalan 10.000 ulangan benih B.SEED.
(b) Sama, bootstrap blok bergerak atas tahun uji berurutan (panjang blok 3, melingkar, 12 tahun → 4 blok per
ulangan), kabupaten diambil ulang seperti dua tahap, 10.000 ulangan benih B.SEED.
(c) VIIRS: setiap sel tabel L1 `Hasil_Patokan_17` (rata-rata rasio dan satu rasio), dua tahap dengan lima benih
yang sama dan 10.000 ulangan; dicetak sel yang vonis dua tahapnya berbeda antar-benih. Di `patokan_tingkat_m.py`.

**M3 Fase ENSO/IOD (deskriptif, tanpa aturan).** ONI musim AMJ (berakhir Juni, diketahui sebelum musim kebakaran
Agustus–Oktober), dari `oni.ascii.txt` NOAA: El Niño ≥ +0,5, La Niña ≤ −0,5, selain itu netral. DMI rata-rata Mei–Juni
dari `dmi.had.long.data` (HadISST, NOAA PSL): positif ≥ +0,4, negatif ≤ −0,4. Untuk tiap tahun 2008–2025:
fase, prevalensi, BSS GB atas EF-L dan EF-L atas kalender per tahun (MODIS maju dari log fase K, VIIRS dari log
fase L), rata-rata per fase dengan n. Juga ONI ASO tiap tahun, untuk memeriksa nilai yang disebut naskah.

**R41 (kekeringan; ditulis sebelum dijalankan).**
(a) Bila EF-LD lolos atas EF-L pada suatu sasaran (rata-rata rasio, dua tahap dan t95 > 0) dan satu rasionya
selang dua tahap mengecualikan nol, EF-LD ditambahkan sebagai rujukan keempat untuk sasaran itu dalam evaluasi
2026 (D10 diperluas di berkas ini, sebelum data 2026 standard processing terbit), aturan "kalahkan semua
rujukan" untuk sasaran itu menjadi empat, dan abstrak menyebut bahwa kekeringan kabupaten menambah pada hitungan
nasional.
(b) Bila tidak (di kedua sasaran): evaluasi 2026 tetap dengan tiga rujukan; EF-D dan EF-LD ditambahkan ke D10
sebagai kolom yang dilaporkan, bukan rujukan; §6.2 melaporkan EF-D dan EF-LD atas EF-L dengan ketiga agregasi,
dan abstrak atau §7 menyebut dalam satu klausa bahwa kalender yang disesuaikan kekeringan dalam bentuk tetap tidak
melampaui EF-L, sehingga aturan 2026 tidak memuatnya.
(c) Bila EF-D lolos atas EF-L (kekeringan saja lebih baik dari hitungan nasional), itu dilaporkan di abstrak
apa pun hasil (a).
(d) GB atas EF-LD dilaporkan; bila vonis GB atas EF-LD berbeda dari vonis GB atas EF-L (tak terbedakan per
kabupaten dan nasional), itu ditulis di §6.2.

**R42 (galat Monte Carlo dan blok; ditulis sebelum dijalankan).**
(a) Klaim abstrak "nasional, model riwayat kebakaran mengalahkan EF-L dalam prakiraan maju" tetap hanya bila
batas bawah dua tahap rata-rata rasio GB atas EF-L > 0 di kelima benih dan di 10.000 ulangan. Bila satu saja
≤ 0: abstrak, §1, §6.3 dan §9 menulis bahwa selisih nasional tidak kokoh terhadap galat Monte Carlo bootstrap,
dan Tabel 5 memberi tanda pada sel itu. Hal yang sama untuk GB atas kalender dan atas EF-P ("ketiganya").
(b) Bila (a) bertahan tetapi batas bawah bootstrap blok atas EF-L ≤ 0: abstrak tetap memakai klaim nasional
dengan anak kalimat "but not under a block bootstrap over consecutive years"; §6.3 dan §8 melaporkan kedua selang.
Bila batas bawah blok > 0: satu kalimat di §6.3.
(c) VIIRS: sel L1 yang vonis dua tahapnya berubah antar-benih atau pada 10.000 ulangan ditulis di naskah sebagai
"interval touches zero" (tanpa "excludes zero"), dan Suplemen menyebut rentang batas bawahnya. Bila tidak ada
yang berubah: satu kalimat di §4.3 bahwa vonis tak berubah dengan lima benih dan 10.000 ulangan.

**M3 tanpa aturan.** Untuk 2026, D10 diperluas: hasil 2026 juga dilaporkan menurut fase ONI AMJ 2026 dengan
ambang di atas; tidak ada subset tahun menurut prevalensi teramati.

**Tidak dikerjakan di fase M (dicatat sekarang).** Kekeringan pada jalur MODIS maju (DC GFWED dan CHIRPS di repo
hanya 2012–2025; perlu unduh baru) — ditulis sebagai keterbatasan. Selang yang menghormati korelasi spasial
antar-kabupaten tetangga — tetap di §8. Model dengan prakiraan cuaca — tetap di §8.

**Teks (tanpa analisis) dalam putaran yang sama, dari telaah internal keempat belas:** bagian tingkat sebagai fungsi
belahan dalam satu tabel dari angka yang sudah ada (butir 2); EF-L sebagai rujukan yang harus dikalahkan, bukan
keluaran yang ditawarkan (3); motivasi ENSO dipisah dari offset periode (5); judul sebagai temuan (7); ambang
lawan kuota diturunkan (8); busur abstrak tunggal (12); §4.2 desain maju lebih dulu (15); judul dan urutan §6.3
(16); Tabel 5 dua batas (17); hasil LOYO per kabupaten di abstrak/§9 (18). Tabel A1 diberi kolom "aturan
sebelum/sesudah data dilihat" (6).

## Hasil

Skrip ditulis sesudah komit rancangan c13be9d: `patokan_modis_m.py`, `patokan_tingkat_m.py`, `patokan_enso_m.py`,
jalan 8 Nov. Log (gitignored): `DL_FIRE_MODIS/patokan_modis_m_nas.log`,
`DL_FIRE_NASIONAL/penuh/patokan_tingkat_m_{kab,nas}.log`, `DL_FIRE_NASIONAL/penuh/patokan_enso_m.log`. Empat desimal
dari program. Pemeriksaan reproduksi: MODIS maju nasional GB atas EF-L +0,0496, t95 [+0,0095, +0,0897], dua tahap
benih B.SEED [+0,0081, +0,0880] = fase K; EF-L VIIRS yang dihitung ulang sama dengan `ef1_tingkat` tersimpan
(selisih maks 1e-16); sel L1 benih B.SEED = fase L. DC bulan lalu tidak ada yang hilang di baris uji.

### M1 Rujukan kekeringan (VIIRS 2020–2025): (a) rata-rata rasio [dua tahap] t; (b) satu rasio [dua tahap]; (c) median, tahun > 0

| Sel | Per kabupaten (a) | (b) | (c) | Nasional (a) | (b) | (c) |
|---|---|---|---|---|---|---|
| EF-D atas EF-L | −0,1816 [−0,3108, −0,0840] t [−0,3155, −0,0477] | −0,1299 [−0,2600, −0,0602] | −0,1459 0/6 | −0,0964 [−0,1666, −0,0452] t [−0,1638, −0,0290] | −0,0916 [−0,1608, −0,0452] | −0,0830 0/6 |
| EF-LD atas EF-L | +0,0183 [+0,0011, +0,0371] t [−0,0014, +0,0380] | +0,0156 [+0,0017, +0,0318] | +0,0178 5/6 | **+0,0335** [+0,0123, +0,0535] t [+0,0088, +0,0581] | +0,0298 [+0,0097, +0,0518] | +0,0332 6/6 |
| EF-D atas kalender | **+0,2870** [+0,0937, +0,4681] t [+0,0107, +0,5632] | +0,2694 | +0,3229 5/6 | **+0,1646** [+0,0597, +0,2789] t [+0,0137, +0,3156] | +0,1664 | +0,1658 5/6 |
| EF-LD atas kalender | **+0,3856** [+0,1807, +0,5831] t [+0,0942, +0,6771] | +0,3634 | +0,4048 5/6 | **+0,2568** [+0,1383, +0,3873] t [+0,0870, +0,4265] | +0,2591 | +0,2404 6/6 |
| EF-LD atas EF-P | **+0,3541** | +0,3167 | +0,3457 6/6 | **+0,2605** | +0,2621 | +0,2561 6/6 |
| GB atas EF-LD | +0,0498 [−0,0349, +0,1372] t [−0,0588, +0,1584] | +0,0589 [−0,0471, +0,1417] | +0,0524 4/6 | +0,0044 [−0,1362, +0,1203] t [−0,1738, +0,1826] | −0,0088 [−0,1478, +0,1109] | +0,0474 4/6 |
| RF atas EF-LD | +0,0132 [−0,0631, +0,0795] | +0,0387 [−0,0588, +0,1043] | +0,0277 4/6 | +0,0254 [−0,0325, +0,0925] | +0,0169 [−0,0352, +0,0863] | +0,0247 3/6 |
| Riwayat lokal EF-1m + DC/FFMC atas EF-L | −0,3918 [−0,7393, −0,1078] t [−0,8053, +0,0217] | −0,2124 [−0,5957, −0,0093] | −0,2973 1/6 | −0,2436 [−0,3920, −0,1292] t [−0,4143, −0,0730] | −0,2159 [−0,3671, −0,1135] | −0,2159 0/6 |
| EF-1m + cuaca bulan lalu atas EF-L | −0,2023 [−0,4100, −0,0250] t [−0,4579, +0,0533] | −0,0939 [−0,3465, +0,0517] | −0,1783 1/6 | −0,1630 [−0,2608, −0,0918] t [−0,2639, −0,0620] | −0,1470 [−0,2535, −0,0738] | −0,1525 0/6 |

Tebal = lolos (dua tahap dan t > 0). Koefisien EF-LD per lipatan: per kabupaten logit ef1_halus +0,57 s.d. +0,66,
hitungan nasional +0,56 s.d. +0,79, anomali DC +0,70 s.d. +0,72; nasional +0,68 s.d. +0,80 / +0,46 s.d. +0,72 /
+0,65 s.d. +0,68. Koefisien hitungan nasional dalam EF-LD lebih kecil daripada dalam EF-L (per kabupaten +0,80
s.d. +1,03): kekeringan kabupaten mengambil sebagian peran hitungan nasional.

### M2 Galat Monte Carlo dan ketergantungan antar-tahun

**MODIS maju nasional** (rentang atas lima benih × 1.000 dan 10.000 ulangan; blok bergerak panjang 3, 10.000):

| | Rata-rata rasio | Batas bawah dua tahap, enam jalan | Satu rasio, batas bawah, enam jalan | Blok: rata-rata rasio | Blok: satu rasio |
|---|---|---|---|---|---|
| GB atas EF-L | +0,0496 | +0,0024 .. +0,0125 | +0,0138 .. +0,0226 | [+0,0121, +0,0850] | [+0,0217, +0,0858] |
| GB atas kalender | +0,1366 | +0,0798 .. +0,0837 | +0,0818 .. +0,0855 | [+0,0823, +0,1936] | [+0,0861, +0,1935] |
| GB atas EF-P | +0,1449 | +0,0752 .. +0,0793 | +0,0789 .. +0,0830 | [+0,0768, +0,2178] | [+0,0824, +0,2229] |
| EF-L atas kalender | +0,0874 | +0,0236 .. +0,0288 | +0,0276 .. +0,0297 | [+0,0116, +0,1660] | [+0,0179, +0,1626] |

Batas bawah GB atas EF-L bergantung pada benih bootstrap (+0,0024 pada benih 2), tetapi tetap > 0 di keenam jalan.

**VIIRS, sel yang vonis dua tahapnya berubah antar-jalan** (semua sel L1 dan M1 diperiksa; selebihnya tetap):
- riwayat lokal EF-1m atas EF-L per kabupaten, rata-rata rasio: batas atas −0,0120 .. +0,0019 (fase L: −0,0052,
  "mengecualikan nol");
- GB + hitungan nasional atas EF-L per kabupaten, satu rasio: batas bawah −0,0017 .. +0,0056 (fase L +0,0002); rata-rata
  rasio tetap > 0 (+0,0064 .. +0,0142), t tetap memuat nol;
- EF-LD atas EF-L per kabupaten, rata-rata rasio: batas bawah −0,0000 .. +0,0011; t memuat nol.
Sel terdekat yang tidak berubah: RF atas EF-L nasional, rata-rata rasio, batas bawah −0,0055 .. −0,0002 (selalu memuat nol).

### M3 Fase ENSO/IOD sebelum musim (deskriptif)

BSS per tahun, GB atas EF-L dan EF-L atas kalender (2008–2019 MODIS maju; 2020–2025 VIIRS; fase kosong = netral):

| Tahun | ONI AMJ | ONI ASO | DMI MJ | Per kab prev | GB/EF-L | EF-L/kal | Nas prev | GB/EF-L | EF-L/kal |
|---|---|---|---|---|---|---|---|---|---|
| 2008 | −0,85 La Niña | −0,25 | +0,20 | 4,81 % | +0,035 | +0,226 | 6,99 % | −0,011 | +0,081 |
| 2009 | +0,03 | +0,66 | +0,18 | 10,71 % | −0,003 | −0,013 | 11,43 % | −0,003 | −0,003 |
| 2010 | −0,16 | −1,43 | −0,08 | 2,10 % | −0,468 | +0,543 | 3,93 % | +0,084 | +0,258 |
| 2011 | −0,50 La Niña | −0,76 | −0,02 | 6,95 % | +0,032 | +0,048 | 9,99 % | +0,073 | −0,002 |
| 2012 | −0,18 | +0,37 | −0,18 | 8,53 % | +0,059 | −0,071 | 10,74 % | +0,067 | −0,022 |
| 2013 | −0,30 | −0,32 | −0,50 negatif | 4,80 % | +0,087 | +0,031 | 7,95 % | +0,131 | −0,079 |
| 2014 | +0,35 | +0,24 | −0,06 | 11,70 % | +0,113 | −0,052 | 12,13 % | +0,065 | +0,018 |
| 2015 | +1,04 El Niño | +2,02 | +0,27 | 15,94 % | +0,089 | +0,101 | 13,72 % | +0,104 | +0,093 |
| 2016 | +0,57 El Niño | −0,42 | −0,28 | 2,27 % | −0,178 | +0,415 | 4,00 % | +0,006 | +0,205 |
| 2017 | +0,35 | −0,23 | +0,48 positif | 2,08 % | −0,583 | +0,427 | 3,95 % | −0,095 | +0,329 |
| 2018 | −0,07 | +0,52 | +0,14 | 5,30 % | +0,025 | +0,055 | 6,96 % | +0,093 | +0,099 |
| 2019 | +0,71 El Niño | +0,31 | +0,57 positif | 9,67 % | +0,054 | −0,013 | 10,27 % | +0,081 | +0,071 |
| 2020 | +0,04 | −0,77 | +0,38 | 3,49 % | −0,061 | +0,451 | 5,29 % | −0,269 | +0,295 |
| 2021 | −0,40 | −0,64 | +0,00 | 1,68 % | +0,077 | +0,624 | 4,32 % | +0,117 | +0,363 |
| 2022 | −0,83 La Niña | −0,87 | −0,23 | 1,07 % | +0,205 | +0,703 | 3,33 % | +0,219 | +0,455 |
| 2023 | +0,46 | +1,50 | +0,55 positif | 11,58 % | +0,171 | −0,051 | 11,43 % | +0,004 | +0,063 |
| 2024 | +0,43 | −0,12 | +0,25 | 3,75 % | −0,046 | +0,200 | 6,58 % | +0,027 | +0,081 |
| 2025 | −0,04 | −0,43 | +0,05 | 2,55 % | +0,053 | +0,327 | 5,82 % | +0,125 | +0,133 |

Rata-rata MODIS maju per fase ONI AMJ, per kabupaten GB atas EF-L: El Niño (2015, 2016, 2019) −0,012; La Niña
(2008, 2011) +0,034; netral (7 tahun) −0,110. Nasional: +0,064 / +0,031 / +0,049. Nilai ONI ASO 2020–2023 yang
disebut naskah (−0,77, −0,64, −0,87, +1,50) cocok dengan berkas NOAA; naskah memakai ASO. Fase sebelum musim
berbeda dari fase musim kebakaran di 2010 (AMJ netral, ASO −1,43), 2016 (AMJ El Niño, ASO −0,42) dan 2023 (AMJ
netral, ASO +1,50). Tiga tahun tersepi maju (2010, 2016, 2017) jatuh di fase AMJ netral, El Niño dan netral: fase
ENSO sebelum musim tidak memisahkannya.

### Vonis

**R41a (nasional)**: EF-LD lolos atas EF-L (+0,0335, dua tahap [+0,0123, +0,0535], t [+0,0088, +0,0581]) dan satu
rasionya mengecualikan nol (+0,0298 [+0,0097, +0,0518]); stabil di keenam jalan bootstrap. Maka untuk sasaran
nasional EF-LD menjadi rujukan keempat evaluasi 2026, aturan nasional menjadi "kalahkan keempatnya", dan abstrak
menyebut bahwa kekeringan kabupaten menambah pada hitungan nasional untuk sasaran nasional. **R41b (per
kabupaten)**: EF-LD tidak lolos (t [−0,0014, +0,0380]; dua tahap rata-rata rasio berubah vonis antar-benih); per
kabupaten evaluasi 2026 tetap tiga rujukan dan EF-LD dilaporkan sebagai kolom. **R41c**: EF-D di bawah EF-L di
kedua sasaran (−0,18 dan −0,10, kedua selang) — kekeringan saja tidak menggantikan hitungan nasional; tidak ada
kalimat abstrak. **R41d**: GB atas EF-LD tak terbedakan di kedua sasaran, sama dengan atas EF-L.

**R42a**: batas bawah dua tahap GB atas EF-L, kalender dan EF-P nasional > 0 di kelima benih dan 10.000 ulangan
→ klaim nasional tetap. **R42b**: batas bawah blok bergerak GB atas EF-L +0,0121 > 0 → satu kalimat di §6.3.
**R42c**: tiga sel VIIRS di atas ditulis "interval touches zero"; Suplemen memberi rentang batas.

**D10 diperluas (pendaftaran, sebelum data 2026 standard processing terbit).** Evaluasi 2026: (i) sasaran nasional
dengan empat rujukan (kalender EF-3m, EF-P, EF-L, EF-LD), aturan kalahkan keempatnya; per kabupaten tiga rujukan,
EF-LD dan EF-D dilaporkan sebagai kolom; EF-LD dipasang dengan aturan M1 di tahun latih 2012–2025. (ii) Hasil juga
dilaporkan menurut fase ONI AMJ 2026 (ambang ±0,5); tanpa subset menurut prevalensi teramati. (iii) Interval dua
tahap dilaporkan dengan lima benih bootstrap {B.SEED, 1, 2, 3, 4}; vonis dua tahap hanya bila kelimanya sepakat.

### Bacaan

Kekeringan kabupaten bulan lalu menambah sedikit pada EF-L: nasional selisihnya kokoh (+0,03), per kabupaten tidak.
Kekeringan saja lebih lemah dari hitungan nasional. Model pembelajaran tetap tak terbedakan dari rujukan yang lebih
kuat ini. Klaim nasional MODIS maju bertahan terhadap galat Monte Carlo dan bootstrap blok, dengan batas bawah tipis
(+0,002 sampai +0,013 menurut benih).
