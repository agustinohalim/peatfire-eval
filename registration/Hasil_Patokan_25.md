# Hasil Patokan 25 — fase S revisi (ambang tahun latih sebagai definisi utama Tabel 2, 4 dan 6)

**10 November 2026.** Lanjutan `Hasil_Patokan_24.md`, untuk `Naskah_Artikel2_Revisi.md`.
Pemicu: telaah internal pra-kirim kedua puluh empat dan kedua puluh lima (catatan telaah tidak dirilis): Tabel 2, 4
dan 6 serta §6.2 memakai ambang dan kabupaten yang dinilai dari label 2012–2025 (termasuk tahun uji), sedangkan
abstrak, Q1 dan protokol 2026 memakai ambang tahun latih; telaah meminta satu definisi.

## Rancangan

**Label.** Per lipatan S1 (tahun uji u = 2020, …, 2025), ambang dihitung ulang dari tahun latih (< u) dengan
`label_ulang` (disalin tanpa perubahan dari `patokan_revisi_ambang.py`: persentil 90 kabupaten dengan batas bawah 10;
persentil 90 gabungan). Kabupaten yang dinilai per kabupaten: yang punya positif di tahun latih lipatan (seperti G2).
Label tahun latih juga dipakai untuk melatih semua model dan menghitung semua rujukan.

**Model dan rujukan.** Semua baris Tabel 2 (S1), Tabel 4 dan Tabel 6 (kolom tahun biasa, S1), dengan spesifikasi,
fitur, hiperparameter dan benih (42) persis seperti skrip asalnya; hanya label (latih dan uji) yang berubah:
- tanpa label (dinilai ulang saja): MC, persistensi, musiman naif, skala rasio, FWI nasional/persentil, DC
  nasional/persentil, FFMC nasional;
- fungsi label: EF-1m, EF-3m, EF-P, EF-L, EF-D, EF-LD;
- dilatih ulang: riwayat lokal MC, riwayat lokal EF-1m, riwayat lokal EF-1m + DC/FFMC, logistik (FWI), logistik
  (ONI, DMI), klimatologi + FWI, GB riwayat api, RF, GB, GB tersetel (grid 18 yang sama, pemilihan pada tahun latih),
  MLP, GB + nasional, GB + nasional + DC. RF, GB, MLP, logistik (FWI) dan logistik (ONI, DMI) diambil dari
  `patokan_prediksi_kepekaan_penuh.csv` yang sudah ada (dilatih pada label ulang, benih 42).

**Keluaran.** (S1) Tabel 2 ulang: AUC-PR rata-rata tahun, keunggulan atas rujukan (EF-1m per kabupaten; rujukan
nasional dipilih seperti aslinya), selang dua tahap (1.000 ulangan, benih 2026) dan selang tahun t, kolom "pooled".
(S2) Tabel 4 ulang: BSS atas kalender, EF-P, EF-L, EF-LD dengan sel yang sama, dua tahap kabupaten dan selang tahun t;
tanda † dan ‡ dihitung ulang (‡ dengan lima benih bootstrap 2026, 1, 2, 3, 4). (S3) Tabel 6 kolom tahun biasa ulang:
nilai ekonomi relatif keenam aturan peringatan, a = 0,05/0,1/0,2/0,3/0,5. Kolom tahun ekstrem (S2 2014, 2015, 2019)
tidak dihitung ulang dan tetap dengan ambang seluruh panel (dicatat di keterangan).
Pemeriksaan: RF, GB dan EF-L atas kalender sama dengan G2 (Hasil_Patokan_12) dan P (Hasil_Patokan_22).

## Aturan keputusan

- **R56.** Apa pun hasilnya, Tabel 2, 4 dan 6 (kolom tahun biasa) serta §6.1, §6.2 dan §6.5 diganti ke ambang tahun
  latih; tabel seluruh panel pindah ke suplemen sebagai kepekaan. Gambar 3c memakai bagian tingkat ambang tahun latih
  (fase P).
- **R57.** Sebuah vonis "berubah" bila tanda rata-rata berbalik atau status "kedua selang tanpa nol" berubah. Tiap vonis
  yang berubah disebut di teks dengan kedua angka. Bila sebuah kalimat abstrak bergantung padanya, abstrak ditulis
  ulang dan panel 26 dijalankan; bila tidak, tidak.
- **R58.** Tanda § (klaster provinsi, fase R) tidak dihitung ulang untuk tabel baru; keterangan menyebut bahwa batas
  provinsi dihitung pada ambang seluruh panel (Tabel S16h).

## Skrip dan penyimpangan pelaksanaan (dicatat sebelum dijalankan)

`patokan_ambang_s.py <sasaran>` membangun prakiraan (blok salinan ditandai sumber dan barisnya);
`patokan_ambang_s_tabel.py <sasaran>` mencetak S1–S3 dan baris pemeriksaan. Pelaksanaan yang tidak tertulis persis di
rancangan:

1. Baris tahun latih untuk ambang siaga Tabel 6 memakai label lipatan (tahun < u), termasuk tahun dalam yang ditahan;
   ambang tidak dihitung ulang per lipatan dalam.
2. Per kabupaten, himpunan kabupaten di validasi `xgb_setel` dan di baris ambang = punya positif di tahun latih
   lipatan, bukan `ada_positif` seluruh panel.
3. Rujukan nasional Tabel 2 tetap EF-3m (pasangan terdaftar). Pilihan rujukan tahun latih (patokan_revisi_rujukan.py)
   dicetak per lipatan sebagai pemeriksaan C3, tidak diterapkan otomatis.
4. Selang dua tahap Tabel 2: satu aliran acak untuk kedua puluh dua baris (urutan patokan_revisi_tahun.py); baris yang
   aslinya dari skrip terpisah berbeda hanya oleh derau Monte Carlo.
5. Kolom "pooled" dihitung untuk semua baris; naskah tetap "—" untuk baris yang aslinya tanpa kolom itu.
6. Baris luar-lipatan: `log_klim_jeda` tidak dihitung (tidak dipakai); xgb luar-lipatan hanya per kabupaten.
7. Tidak termasuk keluaran: hitungan §6.2 (sembilan belas prakiraan peluang, empat belas di atas EF-P), bagian tingkat
   (sudah fase P), ulangan 10.000. Bila hitungan §6.2 ikut berubah menurut R56, dihitung dari CSV fase S tanpa model
   baru dan dicatat di sini.
8. Sesudah jalan pertama per kabupaten (S1–S3 sudah terbaca): §6.5 dan S14 naskah juga melaporkan GB − EF-1m untuk
   aturan peluang, hit kuota 10 % GB − EF-1m dan selisih ambang terbaik, yang tidak dicetak. Ditambahkan sebagai blok
   D8 di S3 (prakiraan dan tarikan bootstrap yang sama; b21cb57); tabel per kabupaten dijalankan ulang, keluaran lain
   harus sama persis dengan jalan pertama.
9. Sesudah S1 per kabupaten dan nasional terbaca: §6.1 juga melaporkan kesetaraan (margin 0,05, selang 90 %), selisih
   terdeteksi minimum, bobot kejadian dan Holm atas empat kontras konfirmatori. `patokan_ambang_s_setara.py` menyalin
   fungsi itu tanpa perubahan dari patokan_revisi_setara.py dan menjalankannya pada CSV fase S (benih B.SEED). Analisis
   lima benih tambahan §6.1 tidak diulang dan tetap disebut dengan ambang seluruh panel.

## Hasil

Dijalankan 10 Nov 2026 11:52–12:15 (skrip 8feb367; tambahan D8 b21cb57, setara 402448b). Log:
`DL_FIRE_NASIONAL/penuh/patokan_ambang_s_{kab,nas}.log`, `patokan_ambang_s_tabel_{kab,nas}.log`,
`patokan_ambang_s_tabel_kab2.log` (jalan ulang dengan D8), `patokan_ambang_s_setara.log`.

**Data.** Per kabupaten 409 kabupaten, 29.376 baris, 1.101 positif, prevalensi 3,75 % (407–409 kabupaten per lipatan);
nasional 498 kabupaten, 35.856 baris, 1.931 positif, 5,39 %. Sama dengan G2 (Hasil_Patokan_12).

**Pemeriksaan.** BSS atas kalender per kabupaten: RF +0,3835 [+0,2029, +0,5636], GB +0,3929, EF-L +0,3619; nasional RF
+0,2934, GB +0,2725, EF-L +0,2555 — sama dengan G2/P sampai digit terakhir. Keunggulan AUC-PR RF/GB/MLP per kabupaten
+0,031/+0,028/−0,007 dan nasional +0,053/+0,043 = Tabel S6 (kolom kabupaten per lipatan). C3: rujukan yang dipilih di
dalam lipatan sama dengan pasangan Tabel 2 di keenam lipatan untuk kedua sasaran (R10 tidak terpicu).

### S1. Tabel 2 (keunggulan AUC-PR atas rujukan, [dua tahap] [tahun t])

| Model | Per kab, seluruh panel | Per kab, tahun latih | Nasional, seluruh panel | Nasional, tahun latih |
|---|---|---|---|---|
| MC | −0,119 ** | −0,134 ** | −0,209 ** | −0,230 ** |
| EF-3m / EF-1m | EF-3m −0,052 ** | EF-3m −0,035 [−0,074, −0,000] [−0,078, +0,009] | EF-1m −0,023 ** | EF-1m −0,020 [−0,051, +0,004] [−0,048, +0,008] |
| Riwayat lokal, MC | −0,044 (dua tahap saja) | −0,054 ** | −0,014 | −0,019 |
| Riwayat lokal, EF-1m | +0,002 | +0,008 | +0,025 | +0,026 |
| Klimatologi + FWI | −0,064 (dua tahap saja) | −0,077 ** | −0,226 ** | −0,250 ** |
| GB riwayat api | −0,025 | −0,040 | +0,011 | +0,011 |
| RF | +0,043 | +0,031 [−0,021, +0,082] [−0,014, +0,076] | +0,050 ** | +0,053 [+0,013, +0,099] [+0,003, +0,104] ** |
| GB | +0,031 | +0,028 [−0,040, +0,118] [−0,054, +0,110] | +0,038 (dua tahap saja) | +0,043 [+0,005, +0,083] [−0,002, +0,088] (dua tahap saja) |
| GB tersetel | +0,055 | +0,047 [−0,016, +0,114] [−0,016, +0,110] | +0,045 ** | +0,050 [+0,015, +0,087] [+0,010, +0,089] ** |
| MLP | +0,012 | −0,007 | +0,026 | +0,031 [−0,005, +0,070] [−0,008, +0,071] |

** = kedua selang tanpa nol. Baris lain (persistensi, musiman naif, skala rasio, FWI, DC, FFMC, logistik) tetap di
bawah rujukan dengan status yang sama. **Vonis berubah (R57):** per kabupaten EF-3m di bawah EF-1m tidak lagi di kedua
selang; riwayat lokal MC dan klimatologi + FWI kini di bawah EF-1m di kedua selang; nasional EF-1m di bawah EF-3m tidak
lagi di kedua selang (dan setara dengan EF-3m pada margin 0,05). Tidak ada vonis model terlatih yang berubah.

**Uji §6.1 (penyimpangan 9).** Holm atas empat kontras konfirmatori: RF nasional p 0,042 (Holm 0,168), GB nasional
0,056 (0,168), RF per kab 0,138 (0,276), GB per kab 0,418 (0,418); tak satu pun lolos (seluruh panel: 0,058 terkecil;
sama). Kesetaraan (selang 90 % di dalam ±0,05): per kabupaten tak satu pun model setara dengan EF-1m — riwayat lokal
EF-1m +0,008, 90 % dua tahap [−0,046, +0,060] (seluruh panel: setara); RF 90 % [−0,015, +0,073] [−0,004, +0,066], GB
[−0,036, +0,096] [−0,036, +0,092]. Nasional EF-1m setara dengan EF-3m (−0,020, [−0,044, −0,000] [−0,042, +0,002]).
Selisih terdeteksi minimum 80 %: RF 0,061, GB 0,111 per kabupaten (seluruh panel 0,079 dan 0,091). Bobot kejadian: RF
+0,043 [−0,030, +0,087], GB +0,030 [−0,055, +0,085] per kabupaten. GB tersetel nasional lolos (+0,050), per kabupaten
tidak (+0,047). **Vonis berubah:** kesetaraan riwayat lokal EF-1m dengan EF-1m per kabupaten gugur.

### S2. Tabel 4 (BSS [dua tahap], tanda)

Per kabupaten (seluruh panel → tahun latih): kalender Brier 0,0424, laju konstan 0,0384 (lebih baik di 5 dari 6
tahun); EF-L atas kalender +0,376 → +0,362; EF-P +0,076 → +0,082; RF/GB/GB tersetel atas EF-P +0,369/+0,383/+0,376 →
+0,344/+0,354/+0,351 (tebal); MLP atas EF-P **+0,271 tebal → +0,240 †** ([+0,058, +0,429], tahun t [−0,021,
+0,500]); atas EF-L RF +0,030 → +0,022, GB +0,067 → +0,049, GB tersetel +0,065 → +0,045, MLP −0,126 → −0,131; GB +
nasional atas EF-L +0,091 † → +0,063 [−0,005, +0,122] (tanpa tanda); GB + nasional + DC +0,085 † → +0,064 [+0,001,
+0,121] ‡; EF-LD atas EF-L +0,018 ‡ → +0,016 [−0,000, +0,033], tahun t [+0,001, +0,032] (selang tahun tanpa nol,
dua tahap menyentuh nol di kelima benih); riwayat lokal EF-1m atas EF-L −0,147 ‡ → −0,123 ‡; EF-D atas EF-L −0,182 →
−0,152 (tebal).

Nasional: EF-L atas kalender +0,232 → +0,256; EF-LD atas EF-L +0,034 → +0,035 (tebal, [+0,014, +0,057] tahun t
[+0,011, +0,059]); RF atas EF-L +0,058 → +0,059 [+0,008, +0,121] ‡ (tahun t [−0,007, +0,126]); GB +0,037 → +0,030;
GB tersetel +0,069 → +0,016; MLP atas kalender +0,197 → +0,224 † dan atas EF-P +0,200 → +0,227 †; GB + nasional atas
EF-L +0,065 → +0,058; GB + nasional + DC +0,067 → +0,062; laju konstan −0,453 † → −0,434 †; EF-P −0,005 → −0,004.

**Vonis berubah (R57):** satu — MLP atas EF-P per kabupaten (kedua selang → dua tahap saja). Tak ada sel model
terlatih atas EF-L atau EF-LD yang lolos di kedua selang, untuk kedua sasaran.

### S3. Tabel 6, kolom tahun biasa (nilai ekonomi relatif, a = 0,05 / 0,1 / 0,2 / 0,3 / 0,5)

| Aturan | Model | Seluruh panel | Tahun latih |
|---|---|---|---|
| p > a | GB | +0,380 / +0,219 / +0,077 / −0,001 / −0,021 | +0,380 / +0,252 / +0,127 / +0,033 / −0,026 |
| | EF-L | +0,099 / −0,093 / −0,003 / +0,048 / +0,005 | +0,135 / +0,090 / +0,082 / +0,066 / +0,007 |
| | EF-1m | +0,031 / −0,668 / −1,364 / −1,692 / −1,655 | +0,049 / −0,642 / −1,269 / −1,459 / −1,161 |
| Ambang nilai maks. tahun latih | GB | +0,283 / +0,224 / +0,130 / +0,081 / +0,009 | +0,343 / +0,252 / +0,131 / +0,040 / +0,008 |
| | EF-L | +0,228 / +0,164 / +0,093 / +0,017 / +0,003 | +0,311 / +0,202 / +0,090 / +0,073 / +0,007 |
| | EF-1m | +0,011 / −0,140 / −0,307 / −0,048 / 0,000 | +0,046 / −0,102 / +0,031 / +0,014 / −0,012 |
| Ambang tetap 10 % | GB | +0,241 / +0,263 / +0,159 / +0,011 / −0,462 | +0,251 / +0,265 / +0,138 / −0,038 / −0,600 |
| | EF-L | +0,048 / +0,122 / +0,088 / +0,027 / −0,169 | +0,060 / +0,128 / +0,094 / +0,036 / −0,149 |
| | EF-1m | +0,097 / +0,023 / −0,419 / −1,009 / −2,896 | +0,088 / −0,061 / −0,672 / −1,475 / −4,045 |
| Kuota 10 % | GB | +0,004 / −0,047 / −0,556 / −1,242 / −3,439 | +0,011 / −0,065 / −0,626 / −1,375 / −3,772 |
| | EF-L | −0,061 / −0,112 / −0,629 / −1,326 / −3,556 | −0,025 / −0,102 / −0,668 / −1,423 / −3,839 |
| | EF-1m | −0,061 / −0,109 / −0,625 / −1,322 / −3,550 | −0,053 / −0,131 / −0,701 / −1,460 / −3,891 |
| Kuota disetel | GB | +0,009 / −0,341 / −0,113 / −0,161 / −0,557 | −0,034 / −0,159 / −0,028 / −0,161 / −0,585 |
| | EF-L | −0,143 / −0,415 / −0,094 / −0,187 / −0,593 | −0,140 / −0,186 / −0,064 / −0,201 / −0,642 |
| Ambang terbaik (batas) | GB | +0,457 / +0,344 / +0,194 / +0,127 / +0,069 | +0,451 / +0,341 / +0,196 / +0,135 / +0,075 |
| | EF-L | +0,375 / +0,262 / +0,142 / +0,106 / +0,058 | +0,375 / +0,255 / +0,150 / +0,111 / +0,061 |
| | EF-1m | +0,358 / +0,267 / +0,181 / +0,133 / +0,064 | +0,357 / +0,264 / +0,177 / +0,135 / +0,072 |

Kontras (tahun latih): kuota − ambang tetap GB a = 0,1 −0,330 [−0,580, −0,158] (seluruh panel −0,310); kuota disetel −
ambang nilai maks. GB di kelima a −0,377 / −0,411 / −0,159 / −0,201 / −0,594, dua tahap di bawah nol di kelimanya
(a = 0,2 [−0,337, −0,025], tahun t [−0,367, +0,048]); bagian kuota disetel 30 % di a = 0,05, 10–18 % di a = 0,1, 2 % di
a ≥ 0,2. Ambang tetap GB: siaga 5,1 %, tangkap 38 %; 2021 8 %, 2022 15 %. D8: aturan peluang GB − EF-1m +0,331
[+0,070, +0,771] / +0,895 / +1,396 / +1,492 / +1,135, dua tahap di atas nol di kelima a, tahun t tidak; hit kuota GB −
EF-1m +0,060 [−0,061, +0,189] (GB 42 %, EF-1m 36 %); ambang terbaik GB − EF-1m +0,094 / +0,077 / +0,019 / −0,001 /
+0,002. GB − EF-L aturan peluang +0,244 / **+0,163** / +0,045 / −0,033 / −0,033 (a = 0,05 dua tahap [+0,049, +0,548],
tahun t [−0,065, +0,554]; a = 0,1 [+0,042, +0,322], tahun t [+0,003, +0,323]); ambang nilai maks. +0,032 / +0,050 /
+0,041 / −0,033 / +0,001, tak satu pun selang di atas nol.

**Vonis berubah (R57):** satu — GB atas EF-L dengan aturan peluang pada a = 0,1 kini lolos di kedua selang (seluruh
panel: dua tahap saja), karena aturan peluang EF-L sendiri naik ke +0,090 (seluruh panel −0,093). Tetap eksploratif
(baris EF-L, Bagian 4.4). Kalimat §7 "EF-L ... worth less than the better of always and never ... (−0.093)" tidak
berlaku lagi dan ditulis ulang.

## Kesimpulan menurut aturan

- **R56:** Tabel 2, 3, 4 dan 6 (kolom tahun biasa) serta §6.1, §6.2, §6.5 memakai ambang tahun latih; tabel seluruh
  panel ke suplemen (Tabel S6c, S6d, S6e). Angka yang tidak dihitung ulang di fase S (lima benih, hitungan sembilan
  belas prakiraan atas EF-P, koefisien EF-L/EF-LD, satu rasio, gambut, klaster provinsi) diberi label ambang seluruh
  panel di teks.
- **R57:** vonis yang berubah — Tabel 2: EF-3m per kab, riwayat lokal MC per kab, klimatologi + FWI per kab, EF-1m
  nasional (dan kesetaraannya dengan EF-3m), kesetaraan riwayat lokal EF-1m per kab gugur; Tabel 4: MLP atas EF-P per
  kab; Tabel 6: GB atas EF-L aturan peluang a = 0,1. Tak satu pun kalimat abstrak bergantung padanya (abstrak:
  tak ada model terlatih di atas EF-L per kabupaten di kedua selang — tetap; sebagai peringkat tak ada yang terbedakan
  dari kalender per kabupaten — tetap; bagian tingkat 89 %/57 % dari fase P). **Abstrak tetap; panel 26 tidak
  dijalankan.**
- **R58:** tanda § tidak dihitung ulang; keterangan Tabel 4 menyebut batas provinsi dihitung pada ambang seluruh panel.
- Penyimpangan 10 (sesudah hasil, sebelum gambar dibuat): Gambar 3 baris VIIRS di panel (a) dan (b) juga dipindah ke
  ambang tahun latih (log fase S), bukan hanya panel (c), agar sama dengan Tabel 4; `gambar_inti_s.py`.
