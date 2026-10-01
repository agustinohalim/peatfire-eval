/**
 * Lapisan lahan dan manusia per kabupaten — Google Earth Engine (Artikel 2, Percobaan C)
 *
 * Melengkapi kelompok fitur "statis lahan" dan "manusia" di Artikel_2_Rencana bagian E.
 * Kunci penggabung tetap GID_2 dari aset gadm41_IDN_2_wkt yang sudah diunggah untuk CHIRPS.
 *
 * Tiga tabel, masing-masing diekspor ke ASET (Google Drive penuh, 25 September 2026), lalu
 * diunduh dengan Skrip_Unduh_Aset_Earth_Engine.js:
 *
 *   A. statis_kabupaten     : satu baris per kabupaten
 *        luas_km2, elevasi_m (rata-rata), lereng_deg (rata-rata), tutupan_pohon_2000 (%),
 *        penduduk_2020 (jumlah, WorldPop) -> kepadatan dihitung di Python
 *   B. tutupan_tahunan      : satu baris per kabupaten-tahun, MODIS MCD12Q1 IGBP 500 m,
 *        fraksi luas: hutan (kelas 1-5), semak_savana (6-10), lahan_basah (11),
 *        pertanian (12, 14), perkotaan (13)
 *   C. hilang_hutan_tahunan : satu baris per kabupaten-tahun, Hansen v1.13,
 *        fraksi luas daratan kabupaten yang kehilangan tutupan pohon pada tahun itu
 *        (aset hilang_hutan_tahunan_v2; lihat catatan di bagian C tentang versi pertama)
 *
 * Kebocoran: B dan C dipakai dengan JEDA SATU TAHUN saat digabung (bulan di tahun Y memakai
 * nilai Y-1). Kehilangan hutan pada tahun kebakaran sebagian adalah AKIBAT kebakaran itu,
 * jadi nilai tahun berjalan tidak boleh masuk.
 *
 * Sumber (dicek di katalog Earth Engine, 25 September 2026):
 *   UMD/hansen/global_forest_change_2025_v1_13  lossyear 1-25 = 2001-2025, 30 m, CC-BY-4.0
 *   MODIS/061/MCD12Q1                           LC_Type1, 2001-2023, 500 m
 *   WorldPop/GP/100m/pop                        population, country 'IDN', year 2020
 *   USGS/SRTMGL1_003                            elevation, 30 m
 *
 * Yang TIDAK ada di sini: peta gambut. Belum ada set gambut Indonesia di katalog resmi Earth
 * Engine yang lisensinya sudah diperiksa; ditunda (Artikel_2_Rencana bagian C).
 *
 * Pakai: tempel di code.earthengine.google.com, jalankan, lalu tekan Run pada tiga tugas di tab
 * Tasks. Periksa dulu angka contoh yang dicetak di Console.
 */

var PROJECT = 'projects/ganti-dengan-project-anda/assets/';
var kab = ee.FeatureCollection(PROJECT + 'gadm41_IDN_2_wkt').select(['GID_2']);
var TAHUN_LC = ee.List.sequence(2011, 2023);   // MCD12Q1 berhenti 2023; jeda 1 tahun -> cukup s.d. 2024
var TAHUN_HILANG = ee.List.sequence(2011, 2025);
var TITIK = ee.Geometry.Point([0, 0]);          // ekspor ke aset wajib punya geometri

function tanpaGeometri(fc) {
  return fc.map(function (f) { return ee.Feature(TITIK, f.toDictionary()); });
}

// --------------------------------------------------------------------------
// A. Statis
// --------------------------------------------------------------------------
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');
var lereng = ee.Terrain.slope(dem).rename('lereng_deg');
var pohon = ee.Image('UMD/hansen/global_forest_change_2025_v1_13').select('treecover2000');
var stackMean = dem.rename('elevasi_m').addBands(lereng).addBands(pohon.rename('tutupan_pohon_2000'));

var penduduk = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('country', 'IDN'))
  .filter(ee.Filter.eq('year', 2020))
  .mosaic().select('population').rename('penduduk_2020');

var statisMean = stackMean.reduceRegions({
  collection: kab, reducer: ee.Reducer.mean(), scale: 90, tileScale: 4
});
var statis = penduduk.reduceRegions({
  collection: statisMean, reducer: ee.Reducer.sum().setOutputs(['penduduk_2020']), scale: 100, tileScale: 4
}).map(function (f) {
  return f.set('luas_km2', f.geometry().area(100).divide(1e6));
});

// --------------------------------------------------------------------------
// B. Tutupan lahan tahunan (fraksi)
// --------------------------------------------------------------------------
function fraksiTutupan(th) {
  th = ee.Number(th);
  var lc = ee.ImageCollection('MODIS/061/MCD12Q1')
    .filter(ee.Filter.calendarRange(th, th, 'year')).first().select('LC_Type1');
  var img = lc.gte(1).and(lc.lte(5)).rename('hutan')
    .addBands(lc.gte(6).and(lc.lte(10)).rename('semak_savana'))
    .addBands(lc.eq(11).rename('lahan_basah'))
    .addBands(lc.eq(12).or(lc.eq(14)).rename('pertanian'))
    .addBands(lc.eq(13).rename('perkotaan'));
  return img.reduceRegions({ collection: kab, reducer: ee.Reducer.mean(), scale: 500, tileScale: 4 })
    .map(function (f) { return f.set('tahun', th); });
}
var tutupan = ee.FeatureCollection(TAHUN_LC.map(fraksiTutupan)).flatten();

// --------------------------------------------------------------------------
// C. Kehilangan tutupan pohon tahunan (fraksi luas kabupaten)
// --------------------------------------------------------------------------
// lossyear TERSAMARKAN di piksel tanpa kehilangan. Tanpa unmask(0), rata-ratanya menjadi
// "bagian kehilangan 2001-2025 yang terjadi pada tahun itu" (jumlah per kabupaten = 1), bukan
// fraksi luas. Itu yang terjadi pada ekspor pertama 25 September 2026 (hilang_hutan_tahunan);
// ekspor itu dibuang dan diganti _v2. Air dikeluarkan lewat datamask = 1 (daratan).
var hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13');
var lossyear = hansen.select('lossyear').unmask(0).updateMask(hansen.select('datamask').eq(1));
function fraksiHilang(th) {
  th = ee.Number(th);
  return lossyear.eq(th.subtract(2000)).rename('hilang_hutan')
    .reduceRegions({ collection: kab, reducer: ee.Reducer.mean().setOutputs(['hilang_hutan']),
                     scale: 100, tileScale: 4 })
    .map(function (f) { return f.set('tahun', th); });
}
var hilang = ee.FeatureCollection(TAHUN_HILANG.map(fraksiHilang)).flatten();

// --------------------------------------------------------------------------
// Periksa sebelum ekspor: satu kabupaten Kalimantan Tengah (Pulang Pisau, gambut, 2015)
// --------------------------------------------------------------------------
var contohGid = 'IDN.14.12_1';
print('contoh statis:', statis.filter(ee.Filter.eq('GID_2', contohGid)).first().toDictionary());
print('contoh tutupan 2015:', fraksiTutupan(2015).filter(ee.Filter.eq('GID_2', contohGid)).first().toDictionary());
print('contoh hilang hutan 2015:', fraksiHilang(2015).filter(ee.Filter.eq('GID_2', contohGid)).first().toDictionary());
// Pemeriksaan kewajaran: kehilangan tutupan pohon nasional 2016 menurut Hansen/GFW sekitar
// 2-3 juta ha (20.000-30.000 km2). Angka jauh di atas itu berarti penyamaran salah lagi.
var km2_2016 = lossyear.eq(16).multiply(ee.Image.pixelArea()).divide(1e6)
  .reduceRegion({ reducer: ee.Reducer.sum(), geometry: kab.geometry().bounds(), scale: 300,
                  maxPixels: 1e10, tileScale: 4 });
print('kehilangan 2016 seluruh Indonesia, km2 (perkiraan kasar, skala 300 m):', km2_2016);
print('baris diharapkan: statis 502, tutupan', TAHUN_LC.size().multiply(502), ', hilang', TAHUN_HILANG.size().multiply(502));

// --------------------------------------------------------------------------
// Ekspor ke aset
// --------------------------------------------------------------------------
Export.table.toAsset({ collection: tanpaGeometri(statis), description: 'statis_kabupaten',
  assetId: PROJECT + 'statis_kabupaten' });
Export.table.toAsset({ collection: tanpaGeometri(tutupan), description: 'tutupan_tahunan',
  assetId: PROJECT + 'tutupan_tahunan' });
Export.table.toAsset({ collection: tanpaGeometri(hilang), description: 'hilang_hutan_tahunan_v2',
  assetId: PROJECT + 'hilang_hutan_tahunan_v2' });
print('Tiga ekspor disiapkan. Buka tab Tasks dan tekan Run pada ketiganya.');
