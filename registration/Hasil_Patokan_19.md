# Hasil Tolok Ukur Nasional 19 — fase N revisi (rujukan kekeringan dan bagian tingkat pada prakiraan maju MODIS)

**8 November 2026.** Lanjutan `Hasil_Patokan_18.md` (fase M), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim kelima belas atas draf revisi (catatan telaah tidak
dirilis). Tiga butir yang perlu analisis:
1. Kemenangan nasional prakiraan maju MODIS atas EF-L (+0,05) belum diuji terhadap EF-LD, rujukan keempat
   nasional untuk 2026; fase M menulis "tidak ada masukan kekeringan sebelum 2012" padahal GFWED (MERRA-2)
   tersedia sejak 1980 di portal NASA NCCS.
2. Bagian tingkat (87% per kabupaten, 62% nasional) hanya dihitung pada tahun kronologis VIIRS dan MODIS LOYO,
   tidak pada prakiraan maju, satu-satunya rancangan yang prakiraan sungguhan dan belum terlihat saat
   rujukan dipilih.
3. Uji tanda tahunan dilaporkan untuk per kabupaten saja; nasional (9 dari 12 tahun) tidak.


Data baru: GFWED v2.0 MERRA2.CORRECTED bulanan 2002-12 s.d. 2011-12 diunduh dari portal NCCS dengan
`gabung_gfwed_komponen.py --unduh MERRA2.CORRECTED --awal`, diagregasi ke kabupaten dengan cara yang sama
(`--presip MERRA2.CORRECTED --awal`, keluaran `DL_FIRE_NASIONAL/gfwed_komponen_MERRA2.CORRECTED_awal.csv`,
tidak masuk repo). Unduhan dimulai sebelum rancangan ini dikomit; tidak ada angka hasil yang dihitung darinya
sebelum komit.

## Rancangan fase N (ditulis 8 Nov 2026, SEBELUM skrip fase N ditulis atau dijalankan)

Semua pasca-hoc: tahun maju 2008–2019 sudah dinilai di fase J, K dan M. Lipatan R sama persis dengan fase K/M
(latih = tahun MODIS sebelum tahun uji, ambang dan kabupaten dari tahun latih, lantai bawaan 2, GB enam benih
{42, 1, 2, 3, 4, 5}). Skrip baru `patokan_modis_n.py`, satu sasaran per jalan (y_gabungan, y_kabupaten).
Interval: dua tahap 1.000 ulangan benih B.SEED (versi cepat matriks galat fase M) dan t atas tahun. "Lolos" =
batas bawah dua tahap dan batas bawah t keduanya > 0 (rata-rata rasio tahunan). Pemeriksaan reproduksi: GB atas
EF-L, kalender, EF-P dan EF-L atas kalender harus sama dengan fase K/M (empat desimal).

**N1 EF-D dan EF-LD pada prakiraan maju.** DC bulan lalu kabupaten dari gabungan
`gfwed_komponen_MERRA2.CORRECTED_awal.csv` (2002-12–2011-12) dan `gfwed_komponen_MERRA2.CORRECTED.csv`
(2012–2025), geser satu bulan per kabupaten, hilang diisi median kabupaten, log1p. Anomali DC dan kedua bentuk
sama persis dengan `patokan_tingkat_m.py`: anomali = log1p(DC bulan lalu) − rata-rata kabupaten itu pada bulan
kalender sama di tahun latih (baris latih tanpa tahunnya sendiri); EF-D = logistik logit(ef1_halus) + anomali DC;
EF-LD = logistik logit(ef1_halus) + anomali api nasional + anomali DC; C = 1e6, dipasang di baris tahun latih
tiap lipatan. Sel: GB atas EF-LD, GB atas EF-D, EF-LD atas EF-L, EF-D atas EF-L, EF-LD atas kalender, EF-LD atas
EF-P. Tiap sel tiga agregasi: (a) rata-rata rasio tahunan [dua tahap] t95; (b) satu rasio [dua tahap];
(c) median tahun dan banyak tahun > 0. Untuk GB atas EF-LD juga: batas bawah dua tahap dengan lima benih
{B.SEED, 1, 2, 3, 4} × 1.000 dan 10.000 ulangan, dan bootstrap blok (panjang 3, melingkar, 10.000) seperti M2.
Koefisien tiap lipatan dicetak.

**N2 Bagian tingkat pada prakiraan maju (definisi I4 fase I).** Untuk GB (enam benih; tiap benih digeser
sendiri, Brier dirata-rata atas benih) atas kalender (EF pasangan) dan atas EF-P: (i) BSS mentah; (ii) offset
periode: tiap ramalan digeser satu konstanta logit yang sama untuk semua baris uji 2008–2019 agar rata-rata
gabungannya = laju teramati gabungan; (iii) geser per tahun ke laju teramati tahun itu. Bagian periode =
1 − BSS(ii)/BSS(i), bagian tahun = (BSS(ii) − BSS(iii))/BSS(i), total = jumlahnya. Untuk rata-rata rasio tahunan
(seperti I4) dan satu rasio, dengan selang dua tahap persentil atas ulangan. Juga dicetak konstanta geser
periode (logit) kalender dan GB, dan BSS(iii) GB atas kalender dengan selang (sisa sesudah tingkat).

**N3 Uji tahunan untuk kedua sasaran (deskriptif).** Untuk GB atas EF-L, kalender, EF-P, EF-LD, dan EF-LD atas
EF-L, kedua sasaran: banyak tahun > 0 dari 12, uji tanda dua sisi (binomial p = 0,5), dan uji peringkat bertanda
Wilcoxon dua sisi (eksak, `scipy.stats.wilcoxon`). Tidak menjadi aturan lolos; dilaporkan bersama untuk kedua
sasaran.

**R43 (kekeringan pada MODIS maju; ditulis sebelum dijalankan).**
(a) Bila GB atas EF-LD nasional lolos (rata-rata rasio, dua tahap B.SEED dan t95 > 0) dan batas bawah dua tahap
> 0 di kelima benih dan 10.000 ulangan: abstrak, §6.3 dan §9 menulis bahwa nasional GB mengalahkan keempat
rujukan nasional dalam prakiraan maju.
(b) Bila lolos di B.SEED tetapi tidak di semua benih/10.000: abstrak menulis "beats EF-L with district drought
only at some bootstrap seeds", §6.3 memberi rentang.
(c) Bila tidak lolos di B.SEED: abstrak, §1, §6.3 dan §9 tidak lagi menyebut kemenangan nasional prakiraan maju
sebagai hasil pembanding positif; ditulis "nationally gradient boosting beats EF-L in the forward forecast but is
not distinguishable from EF-L with district drought", dan "kalahkan semua rujukan" nasional pada MODIS maju
dinyatakan tidak terpenuhi.
(d) EF-LD atas EF-L nasional pada MODIS maju dilaporkan di §6.2. Bila tidak lolos, §6.2 menulis bahwa keunggulan
EF-LD atas EF-L di VIIRS tidak terulang pada tahun maju 2008–2019. Aturan evaluasi 2026 (D10 fase M) TIDAK diubah
oleh hasil ini dalam arah mana pun: rujukan 2026 sudah ditetapkan sebelum fase ini, dan menambah atau membuang
rujukan karena hasil MODIS adalah pemilihan pasca-hasil.
(e) Per kabupaten: GB atas EF-LD dan EF-LD atas EF-L dilaporkan di Suplemen; satu kalimat di §6.3. Tidak ada
perubahan klaim per kabupaten kecuali GB lolos atas EF-LD (tidak diharapkan karena GB tidak lolos atas EF-L);
bila itu terjadi, §6.3 menulis dan menjelaskannya.

**R44 (bagian tingkat maju; ditulis sebelum dijalankan).**
(a) Abstrak menyebut bagian total maju per kabupaten dan nasional (rata-rata rasio) dalam satu klausa, apa pun
hasilnya, di samping 87%/62% VIIRS.
(b) Bila bagian total maju per kabupaten < 0,5 (titik): judul temuan "Much of the probability skill … is the
fire level" tidak lagi didukung pada rancangan prakiraan; naskah kembali ke judul deskriptif dan pilihan judul
diserahkan ke penulis; abstrak menulis bahwa bagian tingkat bergantung pada rancangan.
(c) Bila ≥ 0,5: judul tetap; abstrak menyebut bahwa bagian tingkat juga besar pada prakiraan maju.
(d) Bila BSS GB mentah atas kalender maju (rata-rata rasio) ≤ 0, bagian tidak didefinisikan dengan baik; dicetak
dan dilaporkan, R44 (b)/(c) diputuskan pada satu rasio.
(e) Komponen periode dan tahun dilaporkan di satu tabel untuk tiga rancangan (VIIRS kronologis, MODIS LOYO V0,
MODIS maju), angka VIIRS dan V0 dari `Hasil_Patokan_14`.

**Tidak dikerjakan di fase N (dicatat sekarang).** Prakiraan ENSO/IOD sebelum musim (bukan ONI AMJ teramati) —
keterbatasan §8. EF-LD per kabupaten sebagai rujukan 2026 — tidak; R41 sudah memutuskannya.

## Hasil

Skrip ditulis sesudah komit rancangan 9fbe081: `patokan_modis_n.py`, jalan 8 Nov. Log (gitignored):
`DL_FIRE_MODIS/patokan_modis_n_{nas,kab}.log`; galat kuadrat `DL_FIRE_MODIS/sse_modis_n_{nas,kab}.npz`. Empat
desimal dari program. Unduhan GFWED 2002-12–2011-12: 109 berkas, beberapa kali putus SSL (curl 35) dan diulang;
agregasi 54.718 baris (502 kabupaten × 109 bulan). DC bulan lalu tidak ada yang hilang di 137.448 baris panel.

**Pemeriksaan reproduksi (lulus).** Nasional GB atas EF-L +0,0496 [+0,0081, +0,0880] t [+0,0095, +0,0897], atas
kalender +0,1366, atas EF-P +0,1449, EF-L atas kalender +0,0874; per kabupaten GB atas EF-L −0,0615, atas kalender
+0,1270, atas EF-P +0,1023, EF-L atas kalender +0,1414 — semua sama dengan fase K/M, termasuk nilai per tahun.

Koefisien: DC bernilai positif di semua lipatan dan kedua bentuk. Nasional EF-LD 2008 nas +0,446 DC +0,290, 2019
nas +0,558 DC +0,568; per kabupaten 2008 nas +0,503 DC +0,309, 2019 nas +0,578 DC +0,641.

### N1 Rujukan kekeringan, prakiraan maju 2008–2019: (a) rata-rata rasio [dua tahap] t; (b) satu rasio [dua tahap]; (c) median, tahun > 0

| Sel | Per kabupaten (a) | (b) | (c) | Nasional (a) | (b) | (c) |
|---|---|---|---|---|---|---|
| GB atas EF-LD | −0,0972 [−0,2522, +0,0214] t [−0,2563, +0,0619] | −0,0239 [−0,1188, +0,0309] | +0,0120 8/12 | +0,0257 [−0,0144, +0,0607] t [−0,0124, +0,0638] | +0,0310 [−0,0039, +0,0624] | +0,0579 7/12 |
| GB atas EF-D | −0,0389 [−0,1567, +0,0459] t [−0,1575, +0,0797] | +0,0081 [−0,0631, +0,0536] | +0,0216 9/12 | **+0,0635** [+0,0229, +0,1028] t [+0,0242, +0,1029] | +0,0639 [+0,0265, +0,0996] | +0,0733 10/12 |
| EF-LD atas EF-L | **+0,0303** [+0,0103, +0,0542] t [+0,0073, +0,0534] | +0,0313 [+0,0138, +0,0514] | +0,0222 10/12 | +0,0240 [+0,0028, +0,0491] t [−0,0001, +0,0482] | +0,0250 [+0,0058, +0,0470] | +0,0134 9/12 |
| EF-D atas EF-L | −0,0192 [−0,0788, +0,0286] t [−0,0756, +0,0372] | −0,0000 [−0,0423, +0,0321] | +0,0123 7/12 | −0,0183 [−0,0767, +0,0274] t [−0,0749, +0,0383] | −0,0092 [−0,0580, +0,0275] | +0,0035 7/12 |
| EF-LD atas kalender | **+0,1660** [+0,0651, +0,2758] t [+0,0330, +0,2989] | +0,1423 [+0,0541, +0,2487] | +0,0836 10/12 | **+0,1090** [+0,0441, +0,1883] t [+0,0296, +0,1884] | +0,1122 [+0,0479, +0,1902] | +0,0933 9/12 |
| EF-LD atas EF-P | **+0,1446** [+0,0475, +0,2520] t [+0,0170, +0,2722] | +0,1196 [+0,0361, +0,2199] | +0,0663 9/12 | **+0,1160** [+0,0375, +0,2158] t [+0,0157, +0,2163] | +0,1261 [+0,0421, +0,2262] | +0,0711 8/12 |

Tebal: lolos (dua tahap dan t > 0). GB atas EF-LD, batas bawah dua tahap atas lima benih dan 10.000 ulangan:
nasional rata-rata rasio −0,0144 s.d. −0,0097, satu rasio −0,0057 s.d. −0,0022 (tak satu pun > 0); per kabupaten
−0,2723 s.d. −0,2449 dan −0,1199 s.d. −0,1111. Bootstrap blok (panjang 3, 10.000): nasional GB atas EF-LD +0,0257
[−0,0126, +0,0597], satu rasio [−0,0064, +0,0618]; EF-LD atas EF-L [+0,0047, +0,0432], satu rasio [+0,0080,
+0,0414]. Per kabupaten GB atas EF-LD [−0,2272, +0,0209]; EF-LD atas EF-L [+0,0130, +0,0474].

GB atas EF-LD per tahun, nasional: 2008 −0,060, 2009 −0,019, 2010 −0,036, 2011 +0,088, 2012 +0,063, 2013 +0,073,
2014 +0,055, 2015 +0,064, 2016 −0,003, 2017 −0,073, 2018 +0,095, 2019 +0,061.

### N2 Bagian tingkat pada prakiraan maju (GB enam benih; bagian periode / bagian tahun [dua tahap], total)

| Sasaran, agregasi | atas kalender | atas EF-P |
|---|---|---|
| Per kab., rata-rata rasio | 0,31 [0,09, 0,63] / 0,48 [0,24, 0,68], total 0,79 [0,36, 1,26] | 0,35 [0,07, 0,83] / 0,58 [0,37, 0,85], 0,93 [0,46, 1,58] |
| Per kab., satu rasio | 0,22 [−0,03, 0,52] / 0,35 [0,13, 0,52], 0,56 [0,19, 0,95] | 0,21 [−0,09, 0,64] / 0,45 [0,29, 0,65], 0,66 [0,29, 1,19] |
| Nasional, rata-rata rasio | 0,11 [−0,15, 0,29] / 0,48 [0,25, 0,76], 0,59 [0,24, 0,92] | 0,15 [−0,09, 0,31] / 0,54 [0,28, 0,82], 0,69 [0,29, 1,01] |
| Nasional, satu rasio | 0,09 [−0,18, 0,28] / 0,44 [0,24, 0,67], 0,53 [0,21, 0,85] | 0,15 [−0,10, 0,32] / 0,48 [0,26, 0,72], 0,63 [0,26, 0,95] |

BSS GB atas kalender (rata-rata rasio): per kabupaten mentah +0,1270, sesudah offset periode +0,0875, sesudah geser
per tahun +0,0271 [−0,03, 0,07] t [−0,0273, +0,0814]; nasional +0,1366, +0,1221, +0,0561 [0,01, 0,10] t [+0,0080,
+0,1042]. Geser periode (logit): per kabupaten kalender −0,258, EF-P −0,244, GB −0,022; nasional kalender −0,319,
EF-P −0,392, GB −0,169 (VIIRS: per kabupaten EF-1m −1,252, GB −0,398).

### N3 Uji tahunan (12 tahun maju): tahun > 0, uji tanda dua sisi p, Wilcoxon dua sisi eksak W p

| Sel | Per kabupaten | Nasional |
|---|---|---|
| GB atas EF-L | 8/12, 0,388; W 34, 0,733 | 9/12, 0,146; W 14, 0,052 |
| GB atas kalender | 10/12, 0,039; W 3, 0,002 | 11/12, 0,006; W 1, 0,001 |
| GB atas EF-P | 10/12, 0,039; W 4, 0,003 | 11/12, 0,006; W 2, 0,002 |
| GB atas EF-LD | 8/12, 0,388; W 37, 0,910 | 7/12, 0,774; W 21, 0,176 |
| EF-LD atas EF-L | 10/12, 0,039; W 7, 0,009 | 9/12, 0,146; W 14, 0,052 |

## Keputusan menurut aturan yang didaftarkan

**R43 (c) berlaku.** Nasional GB atas EF-LD tidak lolos di B.SEED (+0,0257, dua tahap [−0,0144, +0,0607], t
[−0,0124, +0,0638]) dan di benih mana pun. Abstrak, §1, §6.3 dan §9 tidak lagi menyebut kemenangan nasional
prakiraan maju sebagai hasil pembanding positif; ditulis bahwa nasional GB mengalahkan EF-L, kalender dan EF-P
dalam prakiraan maju tetapi tidak dapat dibedakan dari EF-L dengan kekeringan kabupaten, dan "kalahkan semua
rujukan" nasional pada MODIS maju tidak terpenuhi. GB atas EF-D nasional lolos (+0,0635), dilaporkan di Suplemen.

**R43 (d).** Nasional EF-LD atas EF-L tidak lolos (+0,0240; dua tahap [+0,0028, +0,0491] mengecualikan nol, t
[−0,0001, +0,0482] menyentuh nol; blok [+0,0047, +0,0432]). §6.2 menulis bahwa pada tahun maju keunggulan
nasional EF-LD atas EF-L tidak lolos aturan dua selang, karena selang t tahunan menyentuh nol, walau dua tahap dan
blok mengecualikan nol. Aturan evaluasi 2026 tidak diubah.

**R43 (e).** Per kabupaten EF-LD atas EF-L LOLOS pada prakiraan maju (+0,0303 [+0,0103, +0,0542] t [+0,0073,
+0,0534], 10/12 tahun, blok [+0,0130, +0,0474]); di VIIRS per kabupaten tidak lolos (t [−0,0014, +0,0380]). GB
atas EF-LD per kabupaten tidak terbedakan (−0,0972). Dilaporkan di Suplemen dan satu kalimat §6.3. Rujukan 2026
per kabupaten tidak diubah (R43 (d): penambahan rujukan karena hasil MODIS adalah pemilihan pasca-hasil).

**R44 (c) berlaku.** Bagian total maju per kabupaten 0,79 (rata-rata rasio) ≥ 0,5: judul tetap; abstrak menyebut
bahwa bagian tingkat juga besar pada prakiraan maju (per kabupaten 79%, nasional 59%, di samping 87%/62% VIIRS).
R44 (d) tidak berlaku (BSS mentah > 0). R44 (e): tabel komponen tiga rancangan di Suplemen. Catatan yang harus
masuk naskah: pada prakiraan maju bagian **periode** kecil (0,31 per kabupaten, 0,11 nasional) dan bagian **tahun**
besar (0,48 di kedua sasaran), kebalikan VIIRS (periode 0,60/0,55); jadi "tingkat" pada prakiraan maju terutama
tingkat tiap tahun, bukan offset periode. Satu rasio per kabupaten memberi total 0,56.
