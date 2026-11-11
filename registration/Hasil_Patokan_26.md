# Hasil Patokan 26 — fase T: rujukan keadaan iklim (ONI, DMI) di samping EF-L

**11 November 2026.** Lanjutan `Hasil_Patokan_25.md`, untuk `Naskah_Artikel2_Revisi.md` (dan versi jurnal yang akan
diturunkan darinya). Pemicu: telaah internal pra-kirim ketiga puluh (catatan telaah tidak dirilis). Tiga peran
meminta hal yang sama (editor X3, domain D2, devil's advocate X5): "tingkat api" yang menyerap sebagian besar
keunggulan diambil dari jumlah titik panas nasional bulan lalu, bukan dari keadaan iklim. Tidak ada rujukan yang
menyesuaikan kalender dengan ENSO/IOD yang tertunda, padahal EF-L disamakan dengan rujukan klimatologi-dan-persistensi
(Knaff dan Landsea 1997), yang persistensinya diambil dari keadaan iklim.

**Rancangan ini ditulis dan dikomit sebelum skrip fase T ditulis atau dijalankan.** Belum ada angka fase T yang
dilihat. Yang sudah diketahui sebelum pendaftaran: koefisien dan BSS logistik (ONI, DMI) Tabel 2/4, yang tidak
memakai kalender kabupaten; fase ENSO/IOD per tahun uji (fase M, `patokan_enso_m.py`).

## Pertanyaan

- **T-Q1.** Apakah kalender yang disesuaikan dengan keadaan iklim yang sudah diketahui sebelum bulan sasaran
  menyerap keunggulan model belajar sebanyak EF-L?
- **T-Q2.** Apakah jumlah titik panas nasional bulan lalu membawa informasi di luar ONI dan DMI, atau sebaliknya?
- **T-Q3.** Apakah model belajar unggul atas rujukan iklim itu?

## Rujukan baru (satu bentuk masing-masing, tanpa varian)

Dengan c̃ = EF-1m dihaluskan (sama dengan EF-L) dan regresi logistik tanpa penalti (`C=1e6`), seperti EF-L, EF-D dan
EF-LD:

- **EF-E** (kalender + keadaan iklim): logit P = β₀ + β₁ logit(c̃_dm) + β₂ ONI_m−1 + β₃ DMI_m−1.
- **EF-LE** (EF-L + keadaan iklim): logit P = β₀ + β₁ logit(c̃_dm) + β₂ [log(1 + N_y,m−1) − N̄_m] + β₃ ONI_m−1 + β₄ DMI_m−1.

ONI_m−1 adalah nilai musim tiga bulan yang berakhir pada bulan m−1 (kolom `oni_lag1`, konvensi
`gabung_indeks_nasional.py`: musim ditaruh pada bulan terakhirnya). DMI_m−1 adalah DMI HadISST bulan m−1 (kolom
`dmi_lag1`). Keduanya dipakai mentah, tanpa anomali tahun latih, karena keduanya sudah anomali. Tidak ada interaksi
musim, jeda lain, curah hujan, atau penyetelan. Baris latih dan uji sama dengan EF-L lipatan itu. Kalender baris latih
dihitung dengan meninggalkan tahun baris itu sendiri, sama dengan EF-L.

Nilai yang hilang: bila ONI atau DMI tidak ada untuk sebuah baris latih atau uji, skrip berhenti dan mencatatnya. Tidak
ada pengisian.

## Jalur dan lipatan

1. **VIIRS, S1 2020–2025, ambang tahun latih** (definisi utama sejak fase S), kedua sasaran. Label, kabupaten yang
   dinilai, dan larik Y disalin dari `patokan_ambang_s.py`. Pemeriksaan: EF-L yang dihitung ulang sama dengan kolom
   `ef1_tingkat` di `patokan_ambang_s_<sasaran>.csv` (selisih < 1e-6). Model belajar diambil dari berkas itu tanpa
   dilatih ulang: RF, GB, GB tersetel, MLP, GB + nasional, GB + nasional + DC; juga EF-LD.
2. **MODIS, prakiraan maju 2008–2019** (lipatan R fase N: latih = tahun MODIS sebelum tahun uji, ambang dan kabupaten
   dari tahun latih), kedua sasaran, lantai bawaan. Fungsi label, larik, rujukan dan anomali disalin dari
   `patokan_modis_n.py`. ONI dan DMI dibaca dari berkas yang sama dengan `gabung_indeks_nasional.py`. GB-FH (enam
   benih) dan EF-L, EF-LD diambil dari `DL_FIRE_MODIS/sse_modis_n_<nas|kab>.npz`. Pemeriksaan: SSE EF-L yang dihitung
   ulang sama dengan kolom `efl` di berkas itu (selisih relatif < 1e-6).

Tidak ada jalur lain: tidak ada validasi silang MODIS, tidak ada ambang 2012–2025, tidak ada tahun ekstrem S2.

## Skor dan selang

Sama dengan Tabel 4 (fase S) dan fase N, tanpa perubahan. BSS dalam-tahun, rata-rata rasio tahunan. Dua selang:

- dua tahap kabupaten-tahun, 1.000 ulangan, benih 2026; untuk MODIS urutan rng `dua_tahap_cepat` fase N;
- selang tahun t (df 5 VIIRS, df 11 MODIS).

"Unggul" berarti kedua selang tanpa nol. Satu rasio gabungan dan median tahun dicetak sebagai pelengkap, tanpa vonis.

Sel yang dihitung, untuk tiap jalur dan sasaran:

- **(a) Rujukan:** EF-E atas kalender; EF-E atas EF-L; EF-L atas EF-E; EF-LE atas EF-L; EF-LE atas EF-E.
  Koefisien β per lipatan juga dicetak.
- **(b) Model belajar atas EF-E:**
  - VIIRS: RF, GB, GB tersetel, MLP, GB + nasional, GB + nasional + DC;
  - MODIS: GB-FH.
- **(c) Pelengkap:** EF-LD atas EF-E dan EF-E atas EF-LD.

## Aturan keputusan (sebelum hasil)

- **R59 (T-Q1, T-Q2).** Naskah melaporkan sel (a) dengan angkanya apa pun hasilnya:
  - Bila EF-L atas EF-E unggul per kabupaten di kedua jalur, naskah menulis bahwa jumlah titik panas nasional bulan
    lalu membawa informasi tingkat di luar ONI dan DMI tertunda.
  - Bila EF-E atas EF-L unggul di kedua jalur, naskah menulis sebaliknya. Kalimat abstrak tentang EF-L lalu ditulis
    ulang supaya rujukan iklim disebut lebih dulu.
  - Selain dua keadaan itu (campur antarjalur, atau tak satu pun unggul), naskah menulis bahwa keduanya tak terbedakan
    dengan enam/dua belas tahun.
- **R60 (aturan rujukan tidak bertambah).** EF-E dan EF-LE **tidak** ditambahkan ke aturan "kalahkan semua rujukan"
  makalah ini (R48, empat rujukan) atau ke protokol 2026 yang terdaftar (`Hasil_Patokan_21`). Alasannya: telaah ketiga puluh
  (devil's advocate X2, metode X3) mengkritik rujukan yang ditambahkan sesudah model lolos. Hasil (b) dilaporkan
  sebagai pemeriksaan tambahan:
  - model yang unggul atas EF-E disebut dengan angkanya;
  - model yang tidak unggul atas EF-E tidak menambah klaim negatif.
- **R61 (T-Q3, saran).** Bila EF-LE unggul atas EF-L per kabupaten di kedua jalur, Bagian 7 menyebut EF-LE sebagai
  calon rujukan untuk musim mendatang. Tanpa itu, EF-LE hanya dilaporkan.
- **R62 (tempat).** Hasil fase T masuk satu paragraf di §6.2 (VIIRS) dan §6.3 (MODIS), dengan satu tabel suplemen
  baru (S20). Hasil itu juga masuk paragraf CLIPER di §7. Daftar rujukan §4.2 menambah EF-E dan EF-LE dengan catatan
  "ditambahkan sesudah telaah ketiga puluh, di luar aturan".

## Putaran teks yang menyertai (tanpa analisis baru, dari angka yang sudah ada)

Putaran ini dikerjakan sesudah fase T, apa pun hasilnya:

1. **Judul dan abstrak** dibuat bersyarat pada periode, karena 4 peran telaah ketiga puluh menilainya kritis. Bagian tingkat
   besar terjadi di tahun uji yang lebih sepi daripada tahun latih (2020–2022 La Niña). Di tahun yang lebih ramai,
   bagian itu ≈ 0. Aturan R37d ("the fire level" di judul dan abstrak) tetap.
2. **Vonis menurut aturan sebelum tiap penambahan rujukan** (metode X3, devil's advocate X2). Dibuat sebagai satu
   tabel suplemen dari sel Tabel 4/5 yang sudah ada:
   - aturan "kalender saja";
   - lalu + EF-P;
   - lalu + EF-L;
   - lalu + EF-LD.
3. **Hasil MODIS yang berlawanan arah disebut di teks utama** (metode X1, devil's advocate X4/X8):
   - GB-FH di atas EF-L per kabupaten dalam tiga desain validasi silang;
   - tahun ekstrem 2015/2019.
4. **Periode tahun uji dikaitkan dengan La Niña 2020–2022** (domain D3).

## Skrip

- `patokan_iklim_t.py <sasaran>` (VIIRS) dan `patokan_iklim_t_modis.py <sasaran>` (MODIS). Keduanya ditulis sesudah
  komit ini, dan setiap blok salinan ditandai sumber dan barisnya.
- Log ke `DL_FIRE_NASIONAL/penuh/patokan_iklim_t_{kab,nas}.log` dan `DL_FIRE_MODIS/patokan_iklim_t_modis_{kab,nas}.log`.

## Hasil

Dijalankan 11 Nov 2026 sesudah komit 036a04b. Log: `DL_FIRE_NASIONAL/penuh/patokan_iklim_t_{kab,nas}.log`,
`DL_FIRE_MODIS/patokan_iklim_t_modis_{kab,nas}.log` (tidak masuk repo). Pemeriksaan lolos: EF-L yang dihitung ulang sama
dengan fase S (selisih maks 1e-16) dan fase N (SSE sama persis); ONI/DMI m−1 tidak ada yang hilang. Reproduksi fase N
cocok: GB-FH atas EF-L per kabupaten −0,062, nasional +0,050 (unggul); EF-L atas kalender +0,141 / +0,087.

Koefisien: ONI dan DMI positif di setiap lipatan dan kedua jalur (EF-E per kabupaten VIIRS ONI +0,71 s.d. +0,90, DMI
+0,35 s.d. +0,58). Di EF-LE, koefisien nasional hampir tidak berubah dan ONI turun ke sekitar sepertiganya
(VIIRS per kabupaten +0,24 s.d. +0,30).

Brier rata-rata tahun VIIRS: per kabupaten kalender 0,04242, EF-E 0,03191, EF-L 0,02900, EF-LE 0,02782, EF-LD 0,02859;
nasional kalender 0,03615, EF-E 0,03141, EF-L 0,02719, EF-LE 0,02672, EF-LD 0,02635.

BSS rata-rata rasio tahunan [dua tahap] [tahun t]; **tebal** = kedua selang tanpa nol.

| Sel | VIIRS per kab | VIIRS nasional | MODIS maju per kab | MODIS maju nasional |
|---|---|---|---|---|
| EF-E atas kalender | +0,278 [+0,020, +0,517] [−0,070, +0,625] | +0,126 [−0,096, +0,323] [−0,174, +0,426] | +0,067 [+0,002, +0,146] [−0,017, +0,152] | −0,012 [−0,064, +0,032] [−0,064, +0,040] |
| EF-L atas EF-E | +0,096 [+0,005, +0,216] [−0,052, +0,244] | +0,127 [+0,034, +0,231] [−0,010, +0,265] | +0,081 [−0,010, +0,196] [−0,034, +0,195] | +0,090 [+0,015, +0,187] [−0,007, +0,187] |
| EF-E atas EF-L | −0,133 [−0,328, −0,010] [−0,355, +0,088] | −0,170 [−0,334, −0,041] [−0,376, +0,036] | −0,140 [−0,335, +0,002] [−0,324, +0,045] | −0,136 [−0,308, −0,025] [−0,293, +0,021] |
| EF-LE atas EF-L | +0,030 [−0,006, +0,063] [−0,014, +0,075] | +0,019 [−0,013, +0,047] [−0,022, +0,061] | −0,011 [−0,065, +0,029] [−0,065, +0,043] | −0,012 [−0,042, +0,011] [−0,041, +0,016] |
| EF-LE atas EF-E | **+0,128 [+0,061, +0,224] [+0,017, +0,239]** | **+0,147 [+0,066, +0,232] [+0,034, +0,260]** | +0,083 [+0,019, +0,168] [−0,001, +0,166] | **+0,084 [+0,022, +0,168] [+0,004, +0,165]** |
| EF-LD atas EF-E | +0,111 [+0,023, +0,227] [−0,030, +0,253] | **+0,157 [+0,063, +0,265] [+0,015, +0,298]** | **+0,110 [+0,024, +0,219] [+0,001, +0,219]** | **+0,113 [+0,040, +0,207] [+0,019, +0,206]** |

Satu rasio gabungan EF-L atas EF-E: +0,092 / +0,134 / +0,050 / +0,096; tahun > 0: 4/6, 5/6, 7/12, 9/12.

Model belajar atas EF-E (sel b):

| Model | VIIRS per kab | VIIRS nasional |
|---|---|---|
| RF | **+0,123 [+0,057, +0,211] [+0,019, +0,227]** | **+0,182 [+0,092, +0,272] [+0,060, +0,304]** |
| GB | **+0,146 [+0,052, +0,234] [+0,028, +0,264]** | +0,159 [+0,042, +0,285] [−0,018, +0,336] |
| GB tersetel | **+0,146 [+0,074, +0,230] [+0,045, +0,247]** | +0,146 [−0,004, +0,290] [−0,065, +0,357] |
| MLP | −0,013 [−0,183, +0,141] [−0,230, +0,204] | +0,094 [−0,088, +0,249] [−0,147, +0,336] |
| GB + nasional | **+0,156 [+0,043, +0,258] [+0,012, +0,299]** | **+0,179 [+0,082, +0,296] [+0,027, +0,332]** |
| GB + nasional + DC | **+0,156 [+0,045, +0,259] [+0,015, +0,298]** | **+0,183 [+0,087, +0,294] [+0,037, +0,328]** |

MODIS maju, GB-FH atas EF-E: per kabupaten **+0,058 [+0,012, +0,108] [+0,005, +0,112]** (11/12 tahun > 0); nasional
**+0,141 [+0,079, +0,219] [+0,065, +0,218]** (12/12).

## Keputusan menurut aturan

- **R59: keadaan ketiga, "tak terbedakan".** EF-L atas EF-E tidak unggul dengan kedua selang di jalur mana pun (dua
  tahap tanpa nol di tiga dari empat sel, tahun t selalu dengan nol). EF-E atas EF-L juga tidak (tak satu pun sel).
  Naskah menulis bahwa dengan enam/dua belas tahun keduanya tak terbedakan menurut aturan. Angkanya disebut apa adanya:
  titik EF-L di atas EF-E di keempat sel (+0,08 s.d. +0,13), dan EF-E menutup sekitar dua pertiga jarak kalender ke
  EF-L per kabupaten VIIRS (Brier 0,0424 → 0,0319 → 0,0290). Abstrak tidak berubah karena R59.
- **R60.** Model yang unggul atas EF-E disebut dengan angkanya: per kabupaten VIIRS RF, GB, GB tersetel, GB + nasional,
  GB + nasional + DC; nasional VIIRS RF, GB + nasional, GB + nasional + DC; MODIS maju GB-FH kedua sasaran. Aturan
  "kalahkan semua rujukan" dan protokol 2026 tetap.
- **R61.** EF-LE tidak unggul atas EF-L di jalur mana pun (−0,012 s.d. +0,030). EF-LE hanya dilaporkan; tidak ada
  saran rujukan baru.

Bacaan (bukan aturan): ONI dan DMI tertunda membawa sebagian tingkat tahun, dan model belajar mengalahkan kalender yang
disesuaikan dengan keduanya. Sesudah jumlah titik panas nasional bulan lalu diketahui, keduanya tidak menambah apa-apa.
