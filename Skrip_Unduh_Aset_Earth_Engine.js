/**
 * Mengunduh tabel hasil ekspor aset sebagai CSV, tanpa lewat Google Drive (Drive penuh).
 * Jalankan setelah tugas di tab Tasks berstatus Completed. Klik tiap tautan yang dicetak,
 * lalu simpan CSV-nya ke Riset/Kebakaran_Gambut/DL_FIRE_NASIONAL/.
 *
 * Baris diharapkan: statis_kabupaten 502, tutupan_tahunan 6.526 (13 tahun),
 * hilang_hutan_tahunan_v2 7.530 (15 tahun).
 */

var PROJECT = 'projects/ganti-dengan-project-anda/assets/';
var TABEL = {
  statis_kabupaten: ['GID_2', 'luas_km2', 'elevasi_m', 'lereng_deg', 'tutupan_pohon_2000', 'penduduk_2020'],
  tutupan_tahunan: ['GID_2', 'tahun', 'hutan', 'semak_savana', 'lahan_basah', 'pertanian', 'perkotaan'],
  hilang_hutan_tahunan_v2: ['GID_2', 'tahun', 'hilang_hutan']
};

Object.keys(TABEL).forEach(function (nama) {
  var t = ee.FeatureCollection(PROJECT + nama);
  print(nama + ' baris:', t.size());
  print(t.getDownloadURL({ format: 'CSV', selectors: TABEL[nama], filename: nama }));
});
