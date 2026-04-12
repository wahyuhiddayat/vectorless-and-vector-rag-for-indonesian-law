# BPK JDIH Scraper

Scraper untuk mengakuisisi dokumen peraturan perundang-undangan Indonesia dari situs [JDIH BPK](https://peraturan.bpk.go.id). Menghasilkan metadata (JSON) dan file PDF untuk setiap dokumen.

## Prasyarat

```bash
pip install requests beautifulsoup4
```

## Penggunaan Dasar

```bash
python scraper/bpk_scraper.py [OPTIONS]
```

Laporan Markdown kategori terbaru:

```bash
python scraper/bpk_topk_newest.py [OPTIONS]
```

### Opsi

| Flag | Default | Keterangan |
|------|---------|------------|
| `--jenis` | `8` | Jenis ID peraturan (bisa lebih dari satu) |
| `--pages` | `all` | Halaman yang di-scrape, misal `1-10`, `5`, atau `all` |
| `--output` | `./data` | Direktori output |
| `--delay` | `1.5` | Jeda antar-request (detik) |
| `--skip-pdf` | - | Hanya scrape metadata, skip download PDF |
| `--resume` | - | Lewati dokumen yang sudah pernah di-scrape |
| `--log-level` | `INFO` | Level logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Jenis ID

| ID | Jenis | Jumlah Dokumen |
|----|-------|----------------|
| `8` | UU (Undang-Undang) | ~1923 |
| `9` | Perpu | ~170 |
| `10` | PP (Peraturan Pemerintah) | ~4965 |
| `11` | Perpres (Peraturan Presiden) | ~2640 |
| `42` | PMK (Peraturan Menteri Keuangan) | - |
| `105` | Permenaker (Peraturan Menteri Ketenagakerjaan) | - |
| `182` | Permenkes (Peraturan Menteri Kesehatan) | - |

## Contoh

```bash
# Scrape metadata UU halaman 1-2 (tanpa PDF)
python scraper/bpk_scraper.py --jenis 8 --pages 1-2 --skip-pdf

# Scrape semua UU beserta PDF-nya
python scraper/bpk_scraper.py --jenis 8

# Scrape semua jenis sekaligus
python scraper/bpk_scraper.py --jenis 8 9 10 11

# Lanjutkan scraping yang terputus
python scraper/bpk_scraper.py --jenis 8 --resume

# Scrape halaman tertentu saja
python scraper/bpk_scraper.py --jenis 10 --pages 5

# Debug mode
python scraper/bpk_scraper.py --jenis 8 --pages 1 --skip-pdf --log-level DEBUG

# Buat laporan Markdown top-k dokumen terbaru untuk semua kategori
python scraper/bpk_topk_newest.py --k 3

# Fokus ke satu group
python scraper/bpk_topk_newest.py --group Pusat --k 5

# Fokus ke jenis tertentu dan simpan ke file
python scraper/bpk_topk_newest.py --jenis 8 36 10 --k 10 --output data/reports/bpk_topk_newest.md
```

## Laporan Top-K Terbaru

`bpk_topk_newest.py` membuat ringkasan Markdown dokumen terbaru per kategori
dengan urutan group `Pusat`, `Kementerian/Lembaga`, lalu `Daerah`.

### Opsi

| Flag | Default | Keterangan |
|------|---------|------------|
| `--k` | `3` | Jumlah dokumen terbaru per kategori |
| `--group` | semua | Filter ke satu group besar |
| `--jenis` | semua | Daftar `jenis_id` spesifik; override `--group` |
| `--output` | `data/reports/bpk_topk_newest.md` | Simpan hasil Markdown ke file |
| `--delay` | `1.5` | Jeda antar-request (detik) |
| `--log-level` | `INFO` | Level logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Output

Secara default script selalu menulis ke `data/reports/bpk_topk_newest.md` dan
tidak mencetak isi Markdown ke terminal.

Contoh struktur output file:

```md
# BPK Top-K Newest per Category

## Pusat

### Undang-undang (UU) - 1,923 dokumen

1. [Undang-undang (UU) Nomor 1 Tahun 2026](https://peraturan.bpk.go.id/Details/337869/uu-no-1-tahun-2026) - Penyesuaian Pidana
2. ...
3. ...
```

## Struktur Output

Setiap jenis peraturan memiliki subfolder sendiri. Registry gabungan di root.

```
data/
├── UU/
│   ├── metadata/
│   │   ├── uu-1-2026__337869.json
│   │   └── ...
│   └── pdfs/
│       ├── UU Nomor 1 Tahun 2026.pdf
│       └── ...
├── PP/
│   ├── metadata/
│   └── pdfs/
├── Perpres/
│   ├── metadata/
│   └── pdfs/
├── Perpu/
│   ├── metadata/
│   └── pdfs/
└── registry.json      ← 1 file gabungan semua jenis
```

### Format Nama File Metadata

```
{doc_id}__{detail_id}.json
```

- **doc_id**: `{bentuk_singkat}-{nomor}-{tahun}` (misal `uu-1-2026`, `pp-71-2019`)
- **detail_id**: ID unik dari situs BPK (untuk menghindari duplikasi)

### Contoh Isi Metadata JSON

```json
{
  "detail_id": "337869",
  "slug": "uu-no-1-tahun-2026",
  "url": "https://peraturan.bpk.go.id/Details/337869/uu-no-1-tahun-2026",
  "materi_pokok": "UU ini mengatur mengenai ...",
  "tipe_dokumen": "Peraturan Perundang-undangan",
  "judul": "Undang-undang (UU) Nomor 1 Tahun 2026 tentang Penyesuaian Pidana",
  "kategori": "Pusat",
  "teu": "Indonesia, Pemerintah Pusat",
  "nomor": "1",
  "bentuk": "Undang-undang (UU)",
  "bentuk_singkat": "UU",
  "tahun": "2026",
  "tempat_penetapan": "Jakarta",
  "tanggal_penetapan": "02 Januari 2026",
  "tanggal_pengundangan": "02 Januari 2026",
  "tanggal_berlaku": "02 Januari 2026",
  "sumber": "LN 2026 (1), TLN (7153) : 51 hlm.",
  "subjek": "HUKUM PIDANA, PERDATA, DAN DAGANG",
  "status": "Berlaku",
  "bahasa": "Bahasa Indonesia",
  "lokasi": "Pemerintah Pusat",
  "bidang": "HUKUM PIDANA",
  "relasi": [
    {
      "tipe_relasi": "Mengubah",
      "ref_display": "UU No. 1 Tahun 2023",
      "ref_id": "234935",
      "ref_slug": "uu-no-1-tahun-2023",
      "keterangan": "Kitab Undang-Undang Hukum Pidana"
    }
  ],
  "pdf_files": [
    {
      "file_id": "400929",
      "filename": "UU Nomor 1 Tahun 2026.pdf",
      "href": "/Download/400929/UU%20Nomor%201%20Tahun%202026.pdf"
    }
  ],
  "doc_id": "uu-1-2026"
}
```

### Registry (`registry.json`)

Satu file gabungan berisi ringkasan seluruh dokumen dari semua jenis. Di-generate otomatis di akhir proses dari semua subfolder `*/metadata/`. Setiap entry memiliki field `kategori` (`Pusat`, `Daerah`, `Kementerian/Lembaga`) untuk membedakan asal peraturan.

```json
{
  "uu-1-2026": {
    "doc_id": "uu-1-2026",
    "detail_id": "337869",
    "kategori": "Pusat",
    "bentuk_singkat": "UU",
    "nomor": "1",
    "tahun": "2026",
    "judul": "Undang-undang (UU) Nomor 1 Tahun 2026 tentang Penyesuaian Pidana",
    "status": "Berlaku",
    "tanggal_penetapan": "02 Januari 2026",
    "relasi": [...],
    "has_pdf": true
  }
}
```

## Fitur

- **Resume**: Gunakan `--resume` untuk melanjutkan scraping yang terputus. Scraper akan skip dokumen yang sudah ada berdasarkan `detail_id`.
- **Retry**: Setiap request otomatis di-retry hingga 3 kali jika gagal.
- **Streaming PDF**: PDF di-download secara streaming agar tidak membebani memori.
- **Polite scraping**: Default delay 1.5 detik antar-request.

## Catatan

- Situs BPK menggunakan Cloudflare. Scraper mengandalkan `requests.Session` untuk mempertahankan cookies dari halaman pencarian agar bisa mengakses halaman detail.
- Jika scraper mulai mendapat error 403, coba tingkatkan `--delay`.
- 10 dokumen per halaman pencarian.
