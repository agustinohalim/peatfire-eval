"""
Verifikasi setiap rujukan Naskah_Artikel1.md terhadap Crossref dan arXiv.

Tiga kesalahan sitasi sudah pernah ditemukan pada proyek ini — dua DOI dan satu ID
arXiv — jadi pemeriksaan ini dijalankan atas seluruh daftar, bukan sampel.

Yang diperiksa:
  - bila DOI tercantum: ambil metadatanya, bandingkan judul, jurnal, dan tahun
  - bila DOI tidak ada: cari lewat judul, laporkan DOI kandidat
  - entri arXiv: ambil metadata dari arXiv API

Keluaran: tabel putusan per rujukan. COCOK / JUDUL BEDA / TAHUN BEDA / TIDAK
DITEMUKAN / DOI HILANG. Apa pun selain COCOK harus dibetulkan sebelum kirim.

Pakai:
    python verifikasi_rujukan.py                  # baca Naskah_Artikel1.md
    python verifikasi_rujukan.py references.md    # baca berkas lain

Crossref dan OpenAlex meminta surel kontak agar permintaan tidak dibatasi. Ganti
lewat peubah lingkungan bila berkas ini dipakai orang lain:

    SUREL_KONTAK=anda@contoh.ac.id python verifikasi_rujukan.py
"""

import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.abspath(__file__))
MD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "Naskah_Artikel1.md")
SUREL = os.environ.get("SUREL_KONTAK", "tino.dev@ixteam.asia")
UA = f"verifikasi-rujukan/1.0 (mailto:{SUREL})"


def ambil(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def normal(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return set(w for w in s.split() if len(w) > 2)


def mirip(a, b):
    A, B = normal(a), normal(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(len(A), len(B))


# ------------------------------------------------------------------ baca daftar

teks = io.open(MD, encoding="utf-8").read()
bagian = teks.split("## References", 1)[1]
entri = []
for b in bagian.split("\n"):
    m = re.match(r"^(\d+)\.\s+(.*)$", b.strip())
    if m:
        entri.append((int(m.group(1)), m.group(2)))
print(f"{len(entri)} rujukan dibaca dari {os.path.basename(MD)}\n")


def pecah(s):
    """Ambil judul, tahun, DOI, dan ID arXiv dari satu entri gaya Harvard."""
    doi = None
    m = re.search(r"doi:\s*(10\.\S+?)(?:\s|$|\*)", s)
    if m:
        doi = m.group(1).rstrip(".,;")
    arx = None
    m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", s)
    if m:
        arx = m.group(1)
    tahun = None
    m = re.search(r"\((\d{4})\)", s)
    if m:
        tahun = int(m.group(1))
    # judul: sesudah "(tahun). " sampai titik sebelum nama jurnal bermiring
    judul = None
    m = re.search(r"\(\d{4}\)\.\s*(.+?)(?:\s*\*|\s*arXiv:|\s*doi:|$)", s)
    if m:
        judul = m.group(1).strip().rstrip(".")
    return judul, tahun, doi, arx


hasil = []
for no, s in entri:
    judul, tahun, doi, arx = pecah(s)
    catatan, putusan, temuan = "", "?", ""

    try:
        if arx:
            # tanpa &max_results: arXiv kini menjawab 406 bila parameter itu ada
            url = f"http://export.arxiv.org/api/query?id_list={arx}"
            xml = ambil(url)
            root = ET.fromstring(xml)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            ent = root.find("a:entry", ns)
            if ent is None:
                putusan = "TIDAK DITEMUKAN"
            else:
                jd = " ".join(ent.find("a:title", ns).text.split())
                terbit = ent.find("a:published", ns).text[:4]
                temuan = f"{jd[:70]} | arXiv {terbit}"
                sk = mirip(judul or "", jd)
                if sk < 0.6:
                    putusan, catatan = "JUDUL BEDA", f"kemiripan {sk:.2f}"
                elif tahun and abs(int(terbit) - tahun) > 1:
                    putusan, catatan = "TAHUN BEDA", f"arXiv {terbit} lawan ditulis {tahun}"
                else:
                    putusan = "COCOK"

        elif doi:
            d = json.loads(ambil("https://api.crossref.org/works/" + urllib.parse.quote(doi)))
            m = d["message"]
            jd = (m.get("title") or ["?"])[0]
            jr = (m.get("container-title") or ["?"])[0]
            th = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            temuan = f"{jd[:60]} | {jr[:32]} | {th}"
            sk = mirip(judul or "", jd)
            if sk < 0.6:
                putusan, catatan = "JUDUL BEDA", f"kemiripan {sk:.2f} — DOI menunjuk karya lain"
            elif tahun and th and abs(th - tahun) > 1:
                putusan, catatan = "TAHUN BEDA", f"Crossref {th} lawan ditulis {tahun}"
            else:
                putusan = "COCOK"

        else:
            q = urllib.parse.quote(judul or s)
            d = json.loads(ambil(f"https://api.crossref.org/works?query.bibliographic={q}"
                                 f"&rows=3&mailto={SUREL}"))
            it = d["message"]["items"]
            if not it:
                putusan = "TIDAK DITEMUKAN"
            else:
                best, bs = None, 0.0
                for c in it:
                    sk = mirip(judul or "", (c.get("title") or [""])[0])
                    if sk > bs:
                        best, bs = c, sk
                jd = (best.get("title") or ["?"])[0]
                jr = (best.get("container-title") or ["?"])[0]
                th = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
                temuan = f"{jd[:60]} | {jr[:30]} | {th} | doi:{best.get('DOI')}"
                if bs >= 0.6:
                    putusan = "DOI HILANG"
                    catatan = f"kandidat kemiripan {bs:.2f} — tambahkan DOI ini"
                else:
                    putusan, catatan = "TIDAK DITEMUKAN", f"kemiripan terbaik hanya {bs:.2f}"
    except Exception as e:
        putusan, catatan = "GAGAL AMBIL", str(e)[:60]

    hasil.append((no, judul, putusan, catatan, temuan))
    tanda = "OK  " if putusan == "COCOK" else ">>> "
    print(f"{tanda}[{no:>2}] {putusan:<16} {(judul or '?')[:56]}")
    if catatan:
        print(f"          catatan : {catatan}")
    if putusan != "COCOK" and temuan:
        print(f"          temuan  : {temuan}")
    time.sleep(0.4)

print("\n" + "=" * 78)
print("RINGKASAN")
print("=" * 78)
dari = {}
for _, _, p, _, _ in hasil:
    dari[p] = dari.get(p, 0) + 1
for k, v in sorted(dari.items(), key=lambda x: -x[1]):
    print(f"  {k:<18} {v}")
buruk = [h for h in hasil if h[2] != "COCOK"]
if buruk:
    print(f"\n  {len(buruk)} rujukan perlu dibetulkan sebelum kirim:")
    for no, judul, p, c, _ in buruk:
        print(f"    [{no}] {p} — {(judul or '?')[:52]}")
    sys.exit(1)
print("\n  Semua rujukan terverifikasi.")
