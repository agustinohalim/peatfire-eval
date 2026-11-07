# Hasil Tolok Ukur Nasional 9 — fase D revisi PeatFireBench (kesetaraan, skor proper, FDRS, jeda, gambut, aturan siaga, rezim, tuning)

**7 November 2026.** Lanjutan `Hasil_Patokan_8.md` (fase C), revisi PeatFireBench. Pemicu: telaah internal
pra-kirim kedua atas draf revisi (catatan telaah tidak dirilis); temuan utamanya: "adds little" adalah
klaim kesetaraan dari ketidakmampuan menolak.

## Rancangan fase D (ditulis 7 Nov 2026, SEBELUM skrip mana pun dijalankan)

Data, model, dan prediksi fase A–C tidak berubah kecuali disebut. Benih 42; penamaan rujukan sama
dengan `Hasil_Patokan_8.md` (EF-1m = `klim_frek1`, EF-3m = `klim_frek`, MC = `klimatologi`; pasangan
Tabel 2 = EF-1m per kabupaten, EF-3m nasional). Selang dua tahap seperti fase C (tahun lalu kabupaten,
1.000 ulangan, seed 2026). Kelompok tahun: biasa = S1 2020–2025, ekstrem = S2 2015, 2014, 2019.

**Penggunaan ulang tahun uji.** Tahun 2020–2025 sudah dinilai di fase A, B, dan C dan di naskah yang
terkirim. Fase D tidak membuat tahun uji baru. Karena itu semua hasil fase D yang tidak termasuk
himpunan konfirmatori D2 dibaca sebagai eksploratori, dan naskah menyatakan penggunaan ulang ini di
§4.4 dan Lampiran A. Evaluasi tahun 2026 sebagai data yang belum tersentuh didaftarkan di D10.

### D1. Kesetaraan dan selisih terdeteksi minimum — `patokan_revisi_setara.py`

**Margin kepentingan praktis (ditetapkan sekarang, sebelum melihat hasil D):** ΔAUC-PR = **0,05**
untuk kedua sasaran (per kabupaten ≈ 20 % AUC-PR EF-1m 0,245; nasional dicetak juga relatif terhadap
AUC-PR EF-3m), dan selisih hit rate kuota 10 % = **0,05** (satu dari dua puluh kabupaten-bulan parah
tambahan tertangkap). Alasan: selisih lebih kecil dari itu tidak mengubah daftar kabupaten yang
disiagakan secara berarti pada kuota 10 % dari ±50 kabupaten per bulan; angka ini dipilih tanpa
konsultasi badan mana pun dan dinyatakan begitu di naskah.

Untuk rf, xgb, mlp dan `log_ef1_jeda` terhadap rujukan pasangan, kedua sasaran, tahun biasa:
- selang 90 % dua tahap (persentil) dan selang-t 90 % atas enam keunggulan tahunan (df 5);
- **TOST**: setara bila *kedua* selang 90 % berada di dalam (−0,05, +0,05);
- **selisih terdeteksi minimum** pada daya 80 %, α 0,05 dua sisi: MDD = (t₀,₉₇₅;₅ + t₀,₈₀;₅) · sd/√6,
  sd = simpangan baku enam keunggulan tahunan; dan banyak tahun uji yang dibutuhkan agar keunggulan
  titik terdeteksi dengan daya 80 % (n terkecil dengan (t₀,₉₇₅;ₙ₋₁ + t₀,₈₀;ₙ₋₁) · sd/√n ≤ keunggulan);
- hal yang sama untuk selisih hit rate kuota (GB dan RF atas EF-1m, per kabupaten), memakai per-tahun
  dari `patokan_revisi_siaga.py` yang dihitung ulang di skrip ini.

- **R15 (judul dan abstrak).** (a) Bila model terbaik per kabupaten lolos superioritas (D2) → naskah
  boleh menyatakan keunggulan kecil yang terbedakan dari nol. (b) Bila tidak, dan model terbaik lolos
  TOST → naskah boleh menyatakan keunggulan lebih kecil dari 0,05 AUC-PR. (c) Bila tidak keduanya →
  naskah menyatakan bahwa enam tahun uji tidak dapat memutuskan, menyebut MDD dan banyak tahun yang
  dibutuhkan; kata "little", "adds little", "equivalent" dilarang di judul, abstrak, dan kesimpulan.

### D2. Selang tahun-kecil dan himpunan konfirmatori — `patokan_revisi_setara.py` (skrip sama)

Untuk semua baris Tabel 2 (kedua sasaran, tahun biasa) di samping selang dua tahap: selang-t 95 % atas
enam keunggulan tahunan dan selang jackknife tinggal-satu-tahun (rata-rata ± t₀,₉₇₅;₅ · se_jackknife).
- **Himpunan konfirmatori** = empat kontras: RF dan GB atas EF-1m (per kabupaten), RF dan GB atas
  EF-3m (nasional). Koreksi Holm pada nilai p uji-t tahunan (dua sisi) atas keempatnya. Semua kontras
  lain eksploratori; banyaknya dicetak.
- **R16.** Klaim keunggulan (di mana pun di naskah) hanya bila selang dua tahap 95 % **dan** selang-t
  95 % sama-sama di atas nol; untuk kontras konfirmatori ditambah p Holm < 0,05. Bila GB nasional gagal
  syarat ini, abstrak tidak lagi menyebut GB nasional sebagai menambah keterampilan.
- **Bobot kejadian.** Di samping rata-rata tahun setara: rata-rata keunggulan tahunan berbobot banyak
  positif per tahun, dengan selang dua tahap (bobot dihitung ulang tiap ulangan), dan AUC-PR ternormalisasi
  prevalensi (AP − π)/(1 − π) per tahun. Plot/daftar keunggulan tahunan terhadap prevalensi tahunan.
  Bobot kejadian dilaporkan, tidak menggantikan rata-rata setara sebagai metrik utama (metrik utama
  tidak diubah sesudah melihat hasil).

### D3. Skor proper — `patokan_revisi_proper.py`

Per tahun (lalu rata-rata setara dan berbobot kejadian), kedua sasaran, tahun biasa dan ekstrem:
- **Brier skill score** BSS = 1 − BS_model / BS_rujukan, rujukan pasangan; dan selisih log score
  (log-loss rujukan − log-loss model; peluang dipotong ke [10⁻⁴, 1 − 10⁻⁴], dicatat karena EF bisa 0);
  untuk rf, xgb, mlp, `log_ef1_jeda`; selang dua tahap (biasa) dan selang kabupaten per tahun (ekstrem).
- **Rujukan dikalibrasi ulang** (per kabupaten saja, karena prakiraan luar-lipatan `patokan_revisi_oof.csv`
  hanya ada untuk y_kabupaten, xgb, EF-1m, EF-3m): Platt dari luar-lipatan tahun latih untuk xgb dan
  untuk EF-1m (pada logit EF-1m), lalu BSS xgb-terkalibrasi atas EF-1m-terkalibrasi.
- **Dekomposisi Murphy** (10 bin berjumlah sama per tahun): reliabilitas, resolusi, ketidakpastian
  untuk xgb, rf, EF-1m/EF-3m.
- **Kalibrasi dengan selang** (menjawab temuan ECE): calibration-in-the-large (rata-rata p − prevalensi)
  dan slope kalibrasi (regresi logistik y pada logit p) per tahun, untuk xgb mentah, Platt semua tahun,
  Platt pasca-2015 (definisi C7), dengan selang dua tahap untuk rata-rata tahun biasa; Brier untuk ketiga
  pemetaan.
- **R17.** Bila BSS xgb atau rf atas rujukan lolos syarat R16 sementara AUC-PR tidak → abstrak melaporkan
  keduanya dan menyatakan bahwa keunggulan ada pada peluang, bukan urutan; judul tidak boleh menyatakan
  keunggulan dalam variasi antartahun tanpa kualifikasi "in ranking". Bila BSS negatif dengan rujukan
  mentah tetapi positif dengan rujukan terkalibrasi (atau sebaliknya) → naskah melaporkan keduanya.
  Klaim "rezim berubah" dari Platt pasca-2015 hanya bila selang dua tahap perbaikan Brier atau
  calibration-in-the-large tahun biasa tidak memuat nol; jika memuat → "sugestif".

### D4. Baseline gaya FDRS (DC dan FFMC) — `patokan_revisi_fdrs.py`

FDRS Indonesia membaca Drought Code (potensi asap/gambut) dan FFMC (potensi penyalaan) terhadap kelas
tetap. Data yang ada: rata-rata bulanan DC dan FFMC per kabupaten dari GFWED MERRA2.CORRECTED
(`gfwed_komponen_MERRA2.CORRECTED.csv`), bulan sebelumnya. Baseline baru, tahun latih saja:
- `dc_nasional`: urutkan menurut DC bulan lalu, tanpa pelatihan;
- `dc_persentil`: persentil DC bulan lalu dalam sebaran tahun latih kabupaten itu sendiri;
- `ffmc_nasional`: urutkan menurut FFMC bulan lalu;
- `log_ef1_fdrs`: regresi logistik terskala pada logit EF-1m (tanpa tahun baris sendiri untuk baris latih,
  seperti `log_ef1_jeda`), DC dan FFMC bulan lalu, bulan_sin/cos.
Dibandingkan dengan rujukan pasangan dan dengan baseline FWI komposit (`fwi`, `fwi_persentil`, `klim_fwi`)
di Tabel 2; tahun biasa (dua tahap + selang-t) dan ekstrem (per tahun).
- **Penyimpangan yang dicatat sekarang:** FDRS membaca kelas harian; panel hanya memuat rata-rata
  bulanan, jadi "bagian hari dengan DC di atas ambang kelas" tidak dapat dihitung. Naskah menyebut ini.
- **R18.** Bila baseline DC/FFMC melampaui baseline FWI komposit (lolos R16 dalam selisihnya) → Tabel 2
  memuat baris DC/FFMC dan teks menyatakan bahwa putusan atas cuaca kebakaran bergantung pada indeks;
  bila `log_ef1_fdrs` lolos R16 atas EF-1m → itu dilaporkan sebagai baseline sederhana terkuat bersama
  `log_ef1_jeda`. Bila tidak → baris tetap ditambahkan (Tabel 2 atau suplemen) dan teks menyatakan
  DC/FFMC bulanan juga tidak menambah atas kalender.

### D5. Memisahkan kebakaran bulan lalu dari cuaca bulan lalu — `patokan_revisi_fdrs.py` (skrip sama)

Regresi logistik terskala, tahun latih saja, semuanya dengan logit EF-1m (tanpa tahun baris sendiri) dan
bulan_sin/cos, ditambah:
- `ef1_api`: y bulan lalu (sasaran sama) dan log1p tp_lag1 — kebakaran bulan lalu saja;
- `ef1_cuaca`: DC dan FFMC bulan lalu, hujan_lag1 dan hujan_anom_lag1 — cuaca bulan lalu saja;
- `ef1_keduanya`: gabungan keduanya.
Dan GB (parameter baku) dilatih ulang tanpa kelompok api (tp_lag1..3, tp_lag12): `xgb_tanpa_api`.
- **R19.** Bila `ef1_api` − EF-1m ≥ 0,8 × (`ef1_keduanya` − EF-1m) dan `ef1_cuaca` − EF-1m tidak lolos R16
  → naskah boleh menulis bahwa keunggulan yang tersisa berasal dari kebakaran bulan lalu. Bila cuaca
  bulan lalu menyumbang > 20 % → "data bulan lalu, api dan cuaca". Heading 6.2 mengikuti putusan ini.

### D6. Strata gambut dan provinsi — `patokan_revisi_gambut.py`

Per kabupaten (411 kabupaten dengan positif), Tabel 2 diulang untuk kabupaten bergambut
(`gambut_fraksi` > 0) dan tak bergambut (= 0), dengan kepekaan ambang `gambut_fraksi` ≥ 0,10; model
rf, xgb, mlp, `log_ef1_jeda` atas EF-1m; tahun biasa (dua tahap + selang-t) dan ekstrem (per tahun,
selang kabupaten). Sasaran nasional dihitung pada sub-himpunan yang sama (AUC-PR dalam sub-himpunan)
atas EF-3m. Provinsi: keunggulan xgb atas EF-1m per provinsi (rata-rata tahun biasa, provinsi dengan
≥ 5 kabupaten berpositif), dinamai, dengan banyak kabupaten bergambut di tiap provinsi.
- **R20.** Bila keunggulan di kabupaten bergambut lolos R16 sementara keseluruhan tidak → naskah
  menyatakan hasil per strata di abstrak (null keseluruhan adalah pengenceran). Bila selang strata
  terlalu lebar (lebar > 0,15) → naskah menyatakan bahwa hasil tidak dapat dipastikan untuk kabupaten
  gambut, dan tidak menyiratkan null berlaku untuk kebakaran gambut.

### D7. Aturan siaga pada pijakan sama — `patokan_revisi_siaga2.py`

Untuk xgb, rf dan EF-1m (per kabupaten), tahun biasa dan ekstrem: kuota 10 % per bulan lawan ambang
luar-lipatan (aturan a fase C: ambang yang memberi 10 % siaga pada prakiraan luar-lipatan tahun latih;
EF-1m memakai ambang dari klimatologinya sendiri pada tahun latih) — hit rate, false alarm ratio, bagian
siaga, dan nilai ekonomi relatif (Richardson) pada a = 0,05 / 0,1 / 0,2 / 0,3 / 0,5, dengan selang dua
tahap untuk tahun biasa. Ambang luar-lipatan hanya untuk xgb dan EF-1m (prakiraan luar-lipatan rf tidak
ada); rf memakai kuota saja.
- **R21.** Kata "safer" atau anjuran kuota dihapus kecuali REV kuota ≥ REV ambang pada sebagian besar a
  di **kedua** kelompok tahun. Bila aturan ambang lebih baik di tahun ekstrem, naskah melaporkan
  hit rate ekstrem kedua aturan dan menyatakan pertukarannya.

### D8. Rezim pasca-2015 lawan kekeringan lebih lemah — `patokan_revisi_rezim.py`

Deskriptif (bukan prakiraan): kabupaten dengan `gambut_fraksi` ≥ 0,10 (49 kabupaten) dan, sebagai
kepekaan, > 0 (131); bulan Agustus–Oktober; 2013–2025. Regresi logistik y_kabupaten pada log DC bulan
yang sama, efek tetap bulan, dan indikator pasca-2015 (tahun ≥ 2016); juga regresi linear log1p titik
panas pada peubah yang sama. Selang 95 % koefisien pasca-2015 dari bootstrap gugus tahun (tahun diambil
ulang dengan pengembalian, 1.000, seed 2026). Dicetak juga DC rata-rata Agustus–Oktober per tahun dan
kabupaten bergambut.
- **R22.** Bila koefisien pasca-2015 negatif dengan selang di bawah nol → naskah boleh menulis "less fire
  per unit drought after 2015". Bila tidak → §2, §6.5 dan Limitations menulis bahwa aktivitas kebakaran
  2019 dan 2023 lebih rendah dari 2015 dan bahwa porsi kebijakan dan kekeringan lebih lemah tidak
  dipisahkan; penjelasan "regime change" untuk kalibrasi dihapus.

### D9. GB dengan penyetelan di dalam lipatan — `patokan_revisi_setel.py`

Grid 18 konfigurasi: max_depth {2, 3, 5} × n_estimators {150, 300, 600} × min_child_weight {1, 10};
learning_rate 0,05, subsample dan colsample 0,9, reg_lambda 1, benih 42. Di tiap lipatan (S1 2020–2025,
S2 2015/2014/2019), kedua sasaran: dua tahun latih terakhir (S1) atau dua tahun latih terbaru (S2) jadi
validasi dalam; konfigurasi dipilih menurut AUC-PR dalam-tahun rata-rata di validasi (per kabupaten:
kabupaten berpositif saja); dilatih ulang pada semua tahun latih; dinilai di tahun uji. Tidak ada tahun
uji yang dipakai untuk memilih.
- **R23.** Bila GB tersetel berbeda dari GB baku > 0,01 AUC-PR di salah satu sasaran, atau mengubah
  putusan R15/R16 → Tabel 2 memuat baris GB tersetel dan putusan memakai yang terbaik dari keduanya
  (dilaporkan sebagai pilihan sesudah melihat hasil). Bila tidak → teks menyebut penyetelan tidak
  mengubah hasil. Naskah menyebut "the four models tested", bukan "machine learning" tanpa kualifikasi,
  apa pun hasilnya.

### D10. Evaluasi prospektif 2026 (pendaftaran saja)

Saat data VIIRS standard processing 2026 terbit, protokol fase A–D dibekukan pada rilis kode yang akan
dibuat sebelum naskah dipakai (DOI Zenodo baru, memuat `Hasil_Patokan_6`–`9` sebagai catatan
pendaftaran), lalu tahun 2026 dinilai sekali dengan aturan R8–R23 tanpa perubahan kode. Ambang sasaran
2026 dari tahun 2013–2025. Hasilnya dicatat di berkas terpisah; berkas ini tidak disunting sesudahnya.

### Tidak dikerjakan di fase D (dicatat sekarang)

- Kelas harian DC/FFMC FDRS (butuh GFWED harian per kabupaten; hanya bulanan tersedia).
- Model spasial (tetangga, CNN, GNN) dan model fondasi: di luar cakupan tolok ukur; naskah menyebut
  "the four models tested".
- Pemeriksaan sasaran dengan rekap luas terbakar SiPongi per kabupaten: butuh pengunduhan baru; naskah
  menyebutnya sebagai pemeriksaan yang tersedia.
- Latensi NRT: tetap di Limitations.

## Hasil (dijalankan 7 Nov 2026; komit rancangan `4d9f8de`)

Log di `DL_FIRE_NASIONAL/penuh/patokan_revisi_<nama>.log` (tidak masuk repo). Selang dua tahap
dihitung ulang dengan fungsi `dua_tahap` (bantu) sehingga batasnya bisa berbeda ±0,002 dari fase C
karena urutan tarikan acak berbeda; titik sama.

### D1/D2. Kesetaraan, selang tahun-kecil, konfirmatori (`patokan_revisi_setara.log`)

Catatan metode: untuk rata-rata, jackknife tinggal-satu-tahun memberi galat baku yang persis sama
dengan uji-t, jadi selang jackknife = selang-t (tercetak sama di log). Hanya selang-t dilaporkan.

**Per kabupaten (411 kabupaten, rujukan EF-1m, AP 0,245; margin 0,05 = 20 % relatif).** Prevalensi per
tahun 2020–2025: 3,49 / 1,68 / 1,07 / 11,58 / 3,75 / 2,55 %; positif 172 / 83 / 53 / 571 / 185 / 126.

| Model | Keunggulan | 2 tahap 95 % | t 95 % | 2 tahap 90 % | t 90 % | p | MDD (80 %) | Tahun perlu | Putusan |
|---|---|---|---|---|---|---|---|---|---|
| RF | +0,043 | [−0,013, +0,100] | [−0,015, +0,101] | [−0,003, +0,091] | [−0,003, +0,089] | 0,116 | 0,079 | 16 | tak terputuskan |
| GB | +0,031 | [−0,034, +0,105] | [−0,036, +0,099] | [−0,022, +0,091] | [−0,021, +0,084] | 0,284 | 0,091 | 35 | tak terputuskan |
| MLP | +0,012 | [−0,043, +0,074] | [−0,040, +0,064] | [−0,034, +0,062] | [−0,029, +0,053] | 0,577 | 0,070 | 135 | tak terputuskan |
| Logistik EF-1m + jeda | +0,002 | [−0,057, +0,051] | [−0,053, +0,056] | [−0,049, +0,043] | [−0,041, +0,044] | 0,942 | 0,074 | — | **setara** (TOST) |

Bobot kejadian (bobot = positif per tahun): RF +0,068 [−0,017, +0,121], GB +0,044 [−0,041, +0,099],
MLP +0,010, logistik EF-1m + jeda +0,007. AUC-PR ternormalisasi prevalensi: RF +0,046 [−0,013, +0,106],
GB +0,033 [−0,034, +0,111]; putusan sama. Semua baseline lain di bawah EF-1m; EF-3m −0,052 (t95
[−0,102, −0,002]).

**Nasional (498 kabupaten, rujukan EF-3m, AP 0,688; margin 0,05 = 7 % relatif).**

| Model | Keunggulan | 2 tahap 95 % | t 95 % | p | MDD | Tahun perlu | Putusan |
|---|---|---|---|---|---|---|---|
| RF | +0,050 | [+0,020, +0,086] | [+0,015, +0,085] | 0,015 | 0,048 | 6 | unggul (kedua selang) |
| GB | +0,038 | [+0,007, +0,076] | [−0,001, +0,077] | 0,054 | 0,053 | 10 | tak terputuskan |
| MLP | +0,026 | [−0,009, +0,061] | [−0,009, +0,061] | 0,118 | 0,048 | 16 | tak terputuskan |
| Logistik EF-1m + jeda | +0,025 | [−0,013, +0,070] | [−0,023, +0,072] | 0,237 | 0,064 | 29 | tak terputuskan |

Per tahun RF nasional +0,087 / +0,065 / +0,080 / +0,032 / −0,002 / +0,038; GB +0,070 / +0,046 / +0,081 /
+0,013 / −0,020 / +0,039. EF-1m nasional −0,023 atas EF-3m (setara menurut TOST). Banyak kontras
model-rujukan: 15 per sasaran.

**Himpunan konfirmatori, Holm:** RF nasional p 0,015 → 0,058; GB nasional 0,054 → 0,161; RF per
kabupaten 0,116 → 0,232; GB per kabupaten 0,284. **Tidak satu pun lolos.**

**Kuota 10 % (per kabupaten):** selisih hit rate GB − EF-1m +0,055, 2 tahap 90 % [−0,037, +0,166], t 90 %
[−0,049, +0,160], MDD 0,182, 44 tahun; RF +0,085 [−0,007, +0,208] / [−0,029, +0,200], MDD 0,199, 23 tahun.
Tak terputuskan.

**Putusan R15:** (c). Per kabupaten tidak ada model terbaik yang lolos superioritas maupun TOST. Naskah
menyatakan enam tahun uji tidak dapat memutuskan; MDD 0,079–0,091 AUC-PR; RF butuh ±16 tahun uji dan GB
±35 untuk mendeteksi keunggulan titiknya. Kata "little", "adds little", "equivalent" dilarang di judul,
abstrak, kesimpulan. Yang *boleh* dinyatakan setara: logistik EF-1m + jeda dengan EF-1m (TOST lolos).
**Putusan R16:** nasional, RF lolos kedua selang 95 % tetapi gagal Holm (0,058); GB gagal selang-t.
Abstrak tidak boleh menyatakan GB nasional menambah keterampilan; RF nasional dilaporkan sebagai
keunggulan yang lolos kedua selang tetapi tidak lolos koreksi untuk empat kontras konfirmatori.

### D3. Skor proper (`patokan_revisi_proper.log`)

**D3a. Brier skill score dan selisih log score atas rujukan pasangan mentah, tahun biasa** (2 tahap 95 %;
t 95 %):

| Sasaran, rujukan | Model | BSS | 2 tahap | t | Δ log score | 2 tahap | t |
|---|---|---|---|---|---|---|---|
| Per kab., EF-1m | RF | +0,401 | [+0,225, +0,588] | [+0,141, +0,662] | +0,065 | [+0,038, +0,096] | [+0,026, +0,105] |
| | GB | +0,415 | [+0,230, +0,605] | [+0,147, +0,683] | +0,073 | [+0,044, +0,104] | [+0,032, +0,114] |
| | MLP | +0,312 | [+0,115, +0,533] | [+0,024, +0,600] | +0,059 | [+0,031, +0,091] | [+0,018, +0,100] |
| | Logistik EF-1m + jeda | +0,322 | [+0,159, +0,486] | [+0,091, +0,552] | +0,057 | [+0,034, +0,081] | [+0,026, +0,088] |
| Nasional, EF-3m | RF | +0,271 | [+0,148, +0,424] | [+0,077, +0,465] | +0,036 | [+0,021, +0,053] | [+0,014, +0,058] |
| | GB | +0,255 | [+0,121, +0,423] | [+0,038, +0,473] | +0,038 | [+0,022, +0,060] | [+0,012, +0,065] |
| | MLP | +0,197 | [−0,017, +0,407] | [−0,110, +0,504] | +0,028 | [−0,002, +0,057] | [−0,015, +0,071] |
| | Logistik EF-1m + jeda | +0,248 | [+0,147, +0,358] | [+0,102, +0,393] | +0,036 | [+0,024, +0,049] | [+0,020, +0,052] |

BSS per tahun GB per kab. 2020–2025: +0,418 / +0,653 / +0,764 / +0,129 / +0,163 / +0,363 (bobot kejadian
+0,266). Ekstrem (per kab., GB): 2015 +0,403, 2014 −0,031, 2019 +0,280; nasional GB 2015 +0,434, 2014
+0,223, 2019 +0,313.

Dekomposisi Murphy (10 bin berjumlah sama, rata-rata tahun biasa, per kab.): GB REL 0,0022 RES 0,0091;
RF REL 0,0036 RES 0,0099; EF-1m REL 0,0198 RES 0,0076; UNC 0,0374. Nasional: GB REL 0,0028 RES 0,0225;
EF-3m REL 0,0110 RES 0,0211. **Selisih Brier sebagian besar dari reliabilitas rujukan**: EF-1m adalah
frekuensi dari ≤ 12 tahun latih, kasar dan terlalu tinggi di tahun tenang.

**D3b. Dengan kalibrasi ulang (Platt luar-lipatan, per kab.):** BSS GB-Platt atas EF-1m-Platt +0,311,
2 tahap [+0,168, +0,476], t [+0,103, +0,519]; per tahun +0,226 / +0,459 / +0,627 / +0,286 / +0,091 /
+0,175. Ekstrem 2015 +0,479, 2014 +0,138, 2019 +0,388. Kalibrasi ulang mengurangi Brier EF-1m di tahun
biasa (mis. 2021 0,0508 → 0,0347) tetapi tidak menutup celahnya.

**D3c. Kalibrasi dengan selang (GB, per kab., tahun biasa):** calibration-in-the-large mentah +0,014
[−0,005, +0,030], Platt semua tahun +0,024 [+0,006, +0,041], Platt pasca-2015 +0,003 [−0,018, +0,017];
slope 1,03 [0,92, 1,16] / 1,13 [1,01, 1,27] / 1,11 [1,00, 1,24]. Brier pasca-2015 − Platt semua −0,0011
[−0,0037, +0,0021]; pasca-2015 − mentah −0,0003 [−0,0025, +0,0023]; |CITL| pasca-2015 − Platt semua
−0,012 [−0,024, +0,006]. Ekstrem: CITL GB mentah −0,058 / −0,102 / −0,021 (2015/2014/2019).

**Putusan R17.** BSS dan log score GB dan RF atas rujukan pasangan lolos R16 untuk kedua sasaran,
sementara AUC-PR per kabupaten tidak → abstrak melaporkan keduanya; judul tidak boleh menyatakan
keunggulan dalam variasi antartahun tanpa kualifikasi peringkat. BSS tetap positif sesudah rujukan
dikalibrasi ulang → dilaporkan keduanya. Perbaikan Platt pasca-2015 tidak lolos selang (Brier dan
|CITL| memuat nol) → "sugestif"; penjelasan rezim untuk kalibrasi tidak boleh dinyatakan sebagai temuan.
Karena logistik EF-1m + jeda juga mendapat BSS besar atas EF-1m, keunggulan peluang belum tentu milik
pembelajaran mesin: lihat D3d (pasca-hoc).

### D3d. PASCA-HOC: Brier atas rujukan peluang yang lebih halus (`patokan_revisi_proper2.log`)

**Penyimpangan dari rancangan, dicatat sebelum dijalankan:** ditambahkan sesudah melihat D3a, karena
logistik EF-1m + jeda juga mendapat BSS besar atas EF-1m, sehingga BSS atas EF-1m bisa mengukur kekasaran
rujukan. Hasilnya eksploratori.

Brier rata-rata tahun biasa per kab.: GB 0,0297, RF 0,0303, MLP 0,0357, logistik EF-1m + jeda 0,0340,
EF-3m 0,0440, EF-1m 0,0496. Nasional: RF 0,0303, GB 0,0311, logistik 0,0309, MLP 0,0339, EF-3m 0,0416.

| Sasaran | Perbandingan | BSS | 2 tahap | t | Δ log score [2 tahap] [t] |
|---|---|---|---|---|---|
| Per kab. | GB atas logistik EF-1m + jeda | +0,169 | [+0,069, +0,303] | [+0,013, +0,326] | +0,016 [+0,007, +0,026] [+0,003, +0,029] |
| | RF atas logistik EF-1m + jeda | +0,142 | [+0,067, +0,242] | [+0,021, +0,262] | +0,008 [+0,001, +0,017] [−0,002, +0,019] |
| | MLP atas logistik | −0,000 | [−0,183, +0,205] | [−0,274, +0,274] | +0,002 |
| | GB atas EF-3m | +0,353 | [+0,193, +0,532] | [+0,122, +0,584] | +0,054 (lolos) |
| | logistik atas EF-3m | +0,239 | [+0,110, +0,373] | [+0,066, +0,413] | +0,038 (lolos) |
| Nasional | GB atas logistik | +0,010 | [−0,181, +0,181] | [−0,252, +0,272] | +0,002 |
| | RF atas logistik | +0,036 | [−0,069, +0,154] | [−0,120, +0,193] | −0,000 |

Per tahun BSS GB atas logistik (per kab.): +0,064 / +0,266 / +0,427 / +0,040 / +0,115 / +0,102 (positif
6 dari 6); ekstrem 2015 +0,187, 2014 −0,203, 2019 +0,125. Nasional per tahun −0,425 / +0,159 / +0,291 /
−0,102 / +0,048 / +0,089.

**Bacaan:** (1) kedua kalender (EF-1m, EF-3m) adalah peluang yang buruk; logistik kalender + bulan lalu
sudah memperbaikinya jauh, terutama reliabilitas. (2) Per kabupaten, GB dan RF memberi peluang yang lebih
baik daripada logistik sederhana (BSS lolos kedua selang, positif tiap tahun biasa), padahal urutannya
(AUC-PR) setara. (3) Nasional, tidak ada model yang memberi peluang lebih baik daripada logistik
sederhana. Karena pasca-hoc, (2) dilaporkan sebagai eksploratori dan tidak masuk judul.

### D4. Baseline gaya FDRS (`patokan_revisi_fdrs.log`)

AP per kab. (tahun biasa): DC nasional 0,090, DC persentil 0,091, FFMC nasional 0,122, logistik
EF-1m + DC + FFMC 0,246; FWI komposit 0,124, FWI persentil 0,133, klimatologi + FWI 0,181; EF-1m 0,245.
Nasional: 0,160 / 0,119 / 0,177 / 0,661; FWI 0,198 / 0,140 / 0,462; EF-3m 0,688.

Selisih DC/FFMC − FWI komposit: per kab. DC nasional − FWI −0,034 [−0,062, −0,011] t [−0,067, −0,001];
DC persentil − FWI persentil −0,041 [−0,081, −0,008]; FFMC − FWI −0,002. Nasional −0,038, −0,021, −0,021.
Logistik EF-1m + DC + FFMC − EF-1m (per kab.) +0,001 [−0,043, +0,042]; − klimatologi + FWI +0,065
[−0,006, +0,136] (nasional +0,200, lolos); − logistik EF-1m + jeda −0,000 (per kab.), −0,052 [−0,094,
−0,014] (nasional). Ekstrem: DC/FFMC −0,27 sampai −0,39 per kab. di ketiga tahun; logistik EF-1m + DC +
FFMC +0,004 / −0,085 / +0,017.

**Putusan R18:** DC dan FFMC bulanan tidak melampaui FWI komposit (DC malah lebih rendah); logistik
EF-1m + DC + FFMC tidak lolos atas EF-1m. Baris ditambahkan; teks: DC/FFMC bulanan, dibaca seperti FDRS,
juga tidak menambah atas kalender. Penyimpangan (kelas harian tak dapat dihitung) dinyatakan.

### D5. Kebakaran bulan lalu lawan cuaca bulan lalu (`patokan_revisi_fdrs.log`)

| Model (− rujukan) | Per kab. (EF-1m) | Nasional (EF-3m) |
|---|---|---|
| EF-1m + api bulan lalu | +0,000 [−0,063, +0,055] | +0,026 [−0,012, +0,071] |
| EF-1m + cuaca bulan lalu | +0,004 [−0,028, +0,037] | −0,022 [−0,058, +0,013] |
| EF-1m + keduanya | +0,020 [−0,039, +0,071] | +0,026 [−0,011, +0,069] |
| GB tanpa api | −0,019 [−0,086, +0,048] | −0,085 [−0,121, −0,051] |
| GB − GB tanpa api | +0,050 [+0,025, +0,083] t [+0,035, +0,065] lolos | +0,123 [+0,090, +0,165] lolos |
| GB − EF-1m + keduanya | +0,012 [−0,030, +0,057] | +0,012 [−0,012, +0,038] |

Nasional: keduanya − api +0,001 [−0,011, +0,010]; keduanya − cuaca +0,048 (lolos). Per kab.: keduanya − api
+0,020 [−0,009, +0,040]; keduanya − cuaca +0,015 [−0,030, +0,050].

**Putusan R19:** nasional, api bulan lalu = 97 % dari keunggulan "keduanya" dan cuaca tidak lolos →
"what remains nationally is last month's fire". Per kabupaten, bagian api 0 % dan cuaca 23 % dari
keunggulan yang sendirinya tidak terbedakan dari nol; syarat (a) gagal, cuaca > 20 % → "last month's data,
fire and weather". Di GB, membuang jeda api menurunkan AUC-PR 0,050 (per kab.) dan 0,123 (nasional), lolos.

### D6. Strata gambut dan provinsi (`patokan_revisi_gambut.log`)

| Strata (per kab., EF-1m) | Kab. | AP EF-1m | GB | 2 tahap (lebar) | RF | 2 tahap |
|---|---|---|---|---|---|---|
| Bergambut (> 0) | 125 | 0,186 | +0,072 | [−0,013, +0,169] (0,182) | +0,064 | [−0,023, +0,160] |
| Tak bergambut (= 0) | 286 | 0,276 | +0,014 | [−0,055, +0,104] (0,159) | +0,025 | [−0,029, +0,082] |
| Gambut ≥ 0,10 | 47 | 0,155 | +0,131 | [−0,058, +0,302] (0,360) | +0,109 | [−0,080, +0,276] |

Per tahun GB bergambut +0,032 / +0,080 / +0,145 / +0,171 / −0,059 / +0,063. Ekstrem bergambut (GB, selang
kab.): 2015 +0,132 [+0,086, +0,171], 2014 −0,020 [−0,070, +0,033], 2019 +0,106 [+0,044, +0,166]; tak
bergambut 2015 +0,083, 2014 −0,055 (di bawah nol), 2019 +0,049. Nasional (EF-3m): bergambut RF +0,040
[+0,002, +0,084] t [+0,005, +0,074] (lolos R16), GB +0,031 [−0,008, +0,076]; tak bergambut RF +0,035
[−0,001, +0,081], GB +0,022.

Provinsi (≥ 5 kabupaten berpositif; GB − EF-1m rata-rata tahun dengan positif): GB di atas EF-1m di 15 dari
31 provinsi, RF di 13. Provinsi gambut utama: Jambi +0,158 (8 dari 10 kab. bergambut), Riau +0,032,
Sumatera Selatan +0,017, Kalimantan Tengah +0,011, Kalimantan Selatan +0,015, Kalimantan Barat −0,052;
Papua +0,122. Positif per provinsi kecil (7–116), jadi angka provinsi sangat bising.

**Putusan R20:** keunggulan titik di kabupaten bergambut lebih besar (+0,072 lawan +0,014), tetapi tidak
lolos R16 dan lebar selang > 0,15 → naskah menyatakan hasil tidak dapat dipastikan untuk kabupaten gambut
dan tidak menyiratkan null berlaku untuk kebakaran gambut; perbedaan titik dilaporkan.

### D7. Kuota lawan ambang (`patokan_revisi_siaga2.log`)

| Aturan | Biasa: hit / FAR / siaga | REV a = 0,05 / 0,1 / 0,2 / 0,3 | Ekstrem: hit / FAR / siaga | REV a = 0,05 / 0,1 / 0,2 / 0,3 |
|---|---|---|---|---|
| Kuota GB | 0,405 / 0,868 / 10 % | +0,004 / −0,047 / −0,556 / −1,242 | 0,244 / 0,551 / 10 % | −2,413 / −0,652 / +0,119 / +0,114 |
| Kuota EF-1m | 0,350 / 0,877 / 10 % | −0,061 / −0,109 / −0,625 / −1,322 | 0,228 / 0,578 / 10 % | −2,477 / −0,684 / +0,099 / +0,091 |
| Ambang 10 % GB | 0,365 / 0,658 / 5,1 % | +0,241 / +0,263 / +0,159 / +0,011 | 0,732 / 0,379 / 22,1 % | −0,254 / +0,352 / +0,602 / +0,538 |
| Ambang 10 % EF-1m | 0,407 / 0,790 / 8,9 % | +0,097 / +0,023 / −0,419 / −1,009 | 0,357 / 0,212 / 8,3 % | −1,883 / −0,377 / +0,292 / +0,314 |
| Ambang nilai GB | — | +0,283 / +0,224 / +0,130 / +0,081 | — | +0,367 / +0,461 / +0,568 / +0,442 |

REV kuota − ambang 10 % (GB, biasa, 2 tahap): a 0,05 −0,237 [−0,646, +0,039]; a 0,1 −0,310 [−0,526, −0,136];
a 0,2 −0,714; a 0,3 −1,253; a 0,5 −2,977 (semua di bawah nol kecuali a 0,05). Ekstrem: negatif di ketiga
tahun pada semua a. Hit ambang GB di tahun tenang rendah (2021 0,084, 2022 0,075) dan tinggi di 2023
(0,694); kuota sebaliknya.

**Putusan R21:** kuota kalah dari aturan ambang pada nilai ekonomi di hampir semua a di kedua kelompok
tahun → "safer" dan anjuran kuota dihapus. Naskah melaporkan pertukarannya: ambang menyiagakan lebih banyak
di tahun kebakaran (22 % kabupaten-bulan, hit 0,73 di tahun ekstrem) dan hampir tidak menyiagakan di tahun
tenang.

### D8. Rezim pasca-2015 lawan kekeringan (`patokan_revisi_rezim.log`)

Kabupaten gambut ≥ 0,10 (49), Agustus–Oktober 2013–2025: DC rata-rata 2015 218,6, 2019 149,0, 2023 84,8,
2014 85,6; bagian kabupaten-bulan di atas ambang 0,633 / 0,463 / 0,327 / 0,333. Logistik: log DC +1,772
[+1,347, +2,070]; pasca-2015 −1,425 [−2,477, −0,854] (rasio odds 0,24); linear log1p titik panas pasca-2015
−0,855 [−1,405, −0,333]. Bergambut > 0 (131): pasca-2015 −1,367 [−2,130, −0,948]. 22 dari 1.000 ulangan
dibuang (tanpa tahun pra- atau pasca-2015). Residu per tahun dari model tanpa indikator pasca (y − p dari
DC): 2013 +0,166, 2014 +0,079, 2015 +0,110; 2019 +0,035, 2023 +0,070; tahun tenang pasca-2015 −0,03 sampai
−0,13.

**Putusan R22:** lolos — kebakaran sesudah 2015 lebih rendah per satuan kekeringan (DC) di kabupaten gambut.
Batasan yang wajib disebut: hanya tiga tahun pra-2016; ambang sasaran dihitung dari semua tahun; DC bulan
sama (deskriptif); 2019 dan 2023 tetap di atas garis kekeringan gabungan, jadi pergeseran terutama dari
tahun tenang. Naskah: "less fire per unit drought after 2015 in peat districts", dengan batasan ini;
2019 lebih rendah dari 2015 sebagian karena kekeringan lebih lemah (DC 149 lawan 219).

### D9. GB dengan penyetelan di dalam lipatan (`patokan_revisi_setel.log`)

Konfigurasi terpilih (validasi dua tahun latih terbaru): kedalaman 5 di 14 dari 18 lipatan-sasaran,
600 pohon di 10; kedalaman 2 hanya di lipatan 2020. Kedalaman 5 adalah tepi atas grid, jadi grid mungkin
terlalu sempit; dicatat, tidak diperluas (perluasan sesudah melihat hasil uji = pilihan pasca-hoc).

| Sasaran | AP GB tersetel / baku / rujukan | Tersetel − rujukan [2 tahap] [t] | Tersetel − baku | Tersetel − RF |
|---|---|---|---|---|
| Per kab. (EF-1m) | 0,299 / 0,276 / 0,245 | +0,055 [−0,015, +0,121] [−0,016, +0,125] | +0,023 [−0,013, +0,064] | +0,012 [−0,040, +0,061] |
| Nasional (EF-3m) | 0,733 / 0,726 / 0,688 | +0,045 [+0,015, +0,080] [+0,010, +0,080] **lolos R16** | +0,007 [−0,001, +0,014] | −0,005 |

Per tahun per kab.: +0,019 / +0,096 / +0,109 / +0,093 / −0,066 / +0,076. Ekstrem per kab. 2015 +0,100,
2014 −0,031, 2019 +0,098 (baku +0,102 / −0,046 / +0,071).

**Putusan R23:** GB tersetel berbeda dari GB baku > 0,01 per kabupaten (+0,023) → Tabel 2 memuat baris GB
tersetel. Putusan R15 tidak berubah (per kab. tetap tak terputuskan; RF dan GB tersetel setara titik).
Nasional, GB tersetel lolos kedua selang; karena bukan bagian himpunan konfirmatori, dilaporkan sebagai
eksploratori. Naskah memakai "the models tested".

## Putusan fase D dan akibat untuk naskah

1. **Judul dan tesis (R15, R17).** Dua lensa memberi jawaban berbeda dan keduanya dilaporkan:
   - *Urutan (AUC-PR, metrik utama):* per kabupaten, enam tahun uji tidak dapat memutuskan apakah model yang
     diuji mengurutkan kabupaten-bulan lebih baik daripada kalender kabupaten itu sendiri (selisih
     terdeteksi minimum 0,08–0,09, ±16–35 tahun uji dibutuhkan). Logistik kalender + bulan lalu setara
     dengan kalender (TOST lolos).
   - *Peluang (Brier, log score):* semua model yang diuji, termasuk logistik sederhana, memberi peluang
     yang jauh lebih baik daripada frekuensi kalender (BSS GB +0,41 per kab., +0,26 nasional; tetap +0,31
     sesudah kalender dikalibrasi ulang), terutama karena reliabilitas kalender buruk.
   - Kata "adds little", "little", "equivalent" (untuk model pembelajaran) dilarang. Judul baru menyebut
     kedua lensa, mis. "Six test years cannot tell whether machine learning ranks Indonesian districts'
     fire months better than their own fire calendar, but its probabilities are better calibrated: the
     PeatFireBench benchmark" (redaksi final di naskah).
2. **Nasional (R16).** Tidak ada kontras konfirmatori yang lolos Holm. RF (+0,050) dan GB tersetel (+0,045)
   lolos kedua selang; GB baku gagal selang-t. Abstrak: "random forest and tuned gradient boosting rank
   districts nationally better than the calendar under both intervals, but not after correction for the
   four confirmatory contrasts".
3. **Eksploratori yang boleh disebut di Results, tidak di judul:** per kabupaten GB/RF memberi peluang lebih
   baik daripada logistik sederhana (D3d, pasca-hoc); keunggulan titik lebih besar di kabupaten gambut
   (D6, tak terpastikan).
4. **§6.2 (R19):** nasional, yang tersisa adalah kebakaran bulan lalu; per kabupaten, data bulan lalu (api
   dan cuaca tak terpisahkan). GB kehilangan 0,050 / 0,123 tanpa jeda api.
5. **Cuaca kebakaran (R18):** DC/FFMC bulanan juga tidak menambah atas kalender dan tidak lebih baik dari
   FWI komposit; kelas harian FDRS tidak diuji.
6. **Kalibrasi (R17 bagian Platt):** perbaikan Platt pasca-2015 tidak terbedakan → "suggestive".
7. **Rezim (R22):** "less fire per unit drought after 2015 in peat districts" boleh, dengan batasan tiga
   tahun pra-2016 dan DC bulan sama; 2019 juga lebih kering lemah.
8. **Siaga (R21):** "safer"/anjuran kuota dihapus; aturan ambang menang nilai ekonomi di kedua kelompok
   tahun; pertukaran (tahun tenang hampir tanpa siaga) dilaporkan.
9. **Penggunaan ulang tahun uji** dinyatakan di §4.4 dan Lampiran A; evaluasi 2026 didaftarkan (D10).
10. **Tetap terbuka sebelum naskah dipakai:** ko-penulis diberi tahu (temuan berubah tiga kali), gambar dan
    suplemen dibuat ulang, rilis kode baru dengan DOI (memuat `Hasil_Patokan_6`–`9`).
