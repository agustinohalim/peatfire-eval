/**
 * Indeks cuaca kebakaran (Fire Weather Index System) bulanan per kabupaten — Google Earth Engine
 * PeatFireBench, baseline FWI (Artikel_2_Rencana bagian D, model 8 yang sebelumnya ditunda).
 *
 * Sumber: Global Fire WEather Database (GFWED; Field et al., 2015), perhitungan FWI dari MERRA-2,
 * ~0,5 x 0,625 derajat, harian, 1980 sampai sekarang (jeda ±5 bulan). Aset publik Climate Engine:
 *   projects/climate-engine-pro/assets/ce-merra2_fwi-daily
 * (awesome-gee-community-catalog, dicek 1 Oktober 2026). Data NASA, tanpa batasan pemakaian.
 *
 * Keluaran: satu baris per kabupaten-bulan 2012-01 .. 2025-12 (502 x 168 = 84.336 baris): GID_2,
 * bulan, fwi (rata-rata bulanan), fwi_maks (maksimum harian dalam bulan), fwi_asli (rata-rata
 * tanpa pengisian sel laut; null bila kabupaten tidak menutupi sel daratan), hari. Aset hanya
 * berpita FWI (dicek 1 Oktober 2026); Drought Code dan subindeks lain tidak tersedia.
 *
 * Skala reduksi 5566 m (sama dengan CHIRPS), bukan skala asli ±55 km: dengan skala asli,
 * kabupaten kota yang lebih kecil dari satu sel tidak memuat pusat sel dan mengembalikan null.
 * Pada 5566 m sel diambil tetangga terdekat, jadi kabupaten kecil memakai sel yang menutupinya.
 *
 * Pakai:
 *   1. Tempel di code.earthengine.google.com, tekan Run. Periksa Console: nama pita, contoh
 *      September 2015 (FWI Kalimantan Tengah mestinya tinggi, puluhan), jumlah null.
 *   2. Tab Tasks -> Run pada tugas fwi_bulanan_kabupaten_2012_2025 (ekspor ke ASET, karena Drive
 *      penuh). Perkiraan 10-40 menit.
 *   3. Sesudah Completed: ubah UNDUH menjadi true, Run lagi, klik tautan CSV yang dicetak, simpan
 *      sebagai DL_FIRE_NASIONAL/fwi_bulanan_kabupaten_2012_2025_v2.csv.
 *
 * Kebocoran: FWI bulan berjalan adalah cuaca serentak dengan kebakaran. gabung_fwi_nasional.py
 * hanya memakai jeda (bulan t-1 dst.) untuk model dan baseline prakiraan; FWI bulan berjalan
 * dipakai hanya sebagai rujukan "cuaca sempurna" yang diberi label bukan-prakiraan.
 */

var PROJECT = 'projects/ganti-dengan-project-anda/assets/';
var UNDUH = false;                      // true sesudah tugas ekspor Completed
var NAMA = 'fwi_bulanan_kabupaten_2012_2025_v2';   // v1 = tanpa pengisian sel laut, jangan dipakai
var TAHUN_AWAL = 2012, TAHUN_AKHIR = 2025;
var SKALA = 5566;

if (UNDUH) {
  var t = ee.FeatureCollection(PROJECT + NAMA);
  print('baris (harus 84336):', t.size());
  print('kolom:', t.first().propertyNames());
  print(t.getDownloadURL({ format: 'CSV', filename: NAMA }));
} else {
  var kab = ee.FeatureCollection(PROJECT + 'gadm41_IDN_2_wkt').select(['GID_2']);
  var fwi = ee.ImageCollection('projects/climate-engine-pro/assets/ce-merra2_fwi-daily')
    .filterDate(TAHUN_AWAL + '-01-01', (TAHUN_AKHIR + 1) + '-01-01');
  var pita = fwi.first().bandNames();       // 1 Oktober 2026: hanya ["FWI"]
  var PROJ = fwi.first().projection();
  print('pita aset:', pita);
  print('citra harian 2012-2025:', fwi.size(), '(harus ±5113)');
  print('tanggal terakhir:', ee.Date(fwi.aggregate_max('system:time_start')));

  var bulanList = ee.List.sequence(TAHUN_AWAL, TAHUN_AKHIR).map(function (th) {
    return ee.List.sequence(1, 12).map(function (bl) { return ee.Date.fromYMD(th, bl, 1).millis(); });
  }).flatten();

  var TITIK = ee.Geometry.Point([0, 0]);  // ekspor ke aset wajib punya geometri
  var bulanan = function (ms) {
    var awal = ee.Date(ms), akhir = awal.advance(1, 'month');
    var c = fwi.filterDate(awal, akhir).select('FWI');
    var rata = c.mean(), maks = c.max();
    // GFWED hanya berisi sel daratan MERRA-2; kabupaten pulau dan pesisir sempit jatuh di sel laut
    // (35 null pada contoh 1 Oktober 2026). Sel kosong diisi rata-rata sel daratan dalam 2 sel di
    // sekitarnya, pada proyeksi asli; fwi_asli tetap null di kabupaten itu supaya pengisian
    // dapat ditandai di Python.
    var isi = function (i) { return i.unmask(i.focalMean(2, 'square', 'pixels').reproject(PROJ)); };
    var img = isi(rata).rename('fwi')
      .addBands(isi(maks).rename('fwi_maks'))
      .addBands(rata.rename('fwi_asli'))
      .addBands(ee.Image.constant(c.size()).rename('hari'));
    var label = awal.format('YYYY-MM');
    return img.reduceRegions({ collection: kab, reducer: ee.Reducer.mean(), scale: SKALA, tileScale: 4 })
      .map(function (f) { return ee.Feature(TITIK, f.toDictionary().set('bulan', label)); });
  };

  var contoh = ee.FeatureCollection(bulanan(ee.Date.fromYMD(2015, 9, 1).millis()));
  print('contoh September 2015, lima baris:', contoh.limit(5));
  print('null fwi sesudah diisi (harus 0):', contoh.filter(ee.Filter.notNull(['fwi']).not()).size());
  print('null fwi_asli (sel laut, diisi; sebelumnya 35):', contoh.filter(ee.Filter.notNull(['fwi_asli']).not()).size());
  print('contoh yang diisi:', contoh.filter(ee.Filter.notNull(['fwi_asli']).not()).limit(5));
  print('Kalimantan Tengah (IDN.14.*), rata-rata FWI Sep 2015:',
        contoh.filter(ee.Filter.stringStartsWith('GID_2', 'IDN.14.')).aggregate_mean('fwi'));

  var hasil = ee.FeatureCollection(bulanList.map(bulanan)).flatten();
  Export.table.toAsset({ collection: hasil, description: NAMA, assetId: PROJECT + NAMA });
  print('Ekspor disiapkan: tab Tasks -> Run.');
}
