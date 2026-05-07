# BPK JDIH Scraper

Scraper untuk mengakuisisi dokumen peraturan perundang-undangan Indonesia dari [JDIH BPK](https://peraturan.bpk.go.id). Menghasilkan metadata (JSON) dan file PDF untuk setiap dokumen.

## Prasyarat

```bash
pip install requests beautifulsoup4
```

## Penggunaan Dasar

```bash
python scraper/bpk_scraper.py [OPTIONS]
```

### Opsi

| Flag | Default | Keterangan |
|------|---------|------------|
| `--jenis` | `8` | Jenis ID peraturan, bisa lebih dari satu |
| `--pages` | `all` | Halaman yang di-scrape, misal `1-10`, `5`, atau `all` |
| `--output` | `<project>/data/raw` | Direktori output |
| `--delay` | `1.5` | Jeda antar-request (detik) |
| `--skip-pdf` | - | Hanya scrape metadata, skip download PDF |
| `--tahun` | - | Filter tahun, misal `2024` atau `2020-2026`, range iterasi per tahun |
| `--resume` | - | Lewati dokumen yang sudah pernah di-scrape (cek by `detail_id`) |
| `--skip-doc-ids` | - | Comma-separated doc_id yang dilewati, misal `uu-1-2026,uu-20-2025`. Otomatis di-merge dengan blacklist dari `data/dropped_docs.json` |
| `--limit` | `0` | Maksimum dokumen baru per jenis, `0` artinya tanpa batas |
| `--log-level` | `INFO` | Level logging, `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Jenis ID

Daftar lengkap mapping `jenis_id` ke nama folder ada di `JENIS_MAP` dalam [bpk_scraper.py](bpk_scraper.py). Saat ini ter-cover 28 kategori (4 Pusat, 4 Daerah, 20 Kementerian/Lembaga). Quick reference,

| ID | Folder | Jenis |
|----|--------|-------|
| `8` | `UU` | Undang-Undang |
| `9` | `PERPU` | Peraturan Pemerintah Pengganti UU |
| `10` | `PP` | Peraturan Pemerintah |
| `11` | `PERPRES` | Peraturan Presiden |
| `42` | `PMK` | Peraturan Menteri Keuangan |
| `80` | `PERATURAN_OJK` | Peraturan Otoritas Jasa Keuangan |
| `182` | `PERMENKES` | Peraturan Menteri Kesehatan |

`jenis_id` yang tidak ada di `JENIS_MAP` akan jatuh ke folder fallback `jenis-<id>` dan kategori `Lainnya` (dengan warning di log).

## Contoh

```bash
# Scrape metadata UU halaman 1-2 (tanpa PDF)
python scraper/bpk_scraper.py --jenis 8 --pages 1-2 --skip-pdf

# Scrape semua UU beserta PDF-nya
python scraper/bpk_scraper.py --jenis 8

# Scrape multi-jenis sekaligus
python scraper/bpk_scraper.py --jenis 8 10 11

# Lanjutkan scraping yang terputus
python scraper/bpk_scraper.py --jenis 8 --resume

# Filter tahun, range akan iterasi per tahun
python scraper/bpk_scraper.py --jenis 8 --tahun 2020-2026

# Batasi jumlah dokumen baru per jenis
python scraper/bpk_scraper.py --jenis 8 --limit 50

# Skip beberapa doc_id tertentu
python scraper/bpk_scraper.py --jenis 8 --skip-doc-ids uu-1-2026,uu-20-2025

# Debug mode
python scraper/bpk_scraper.py --jenis 8 --pages 1 --skip-pdf --log-level DEBUG
```

## Struktur Output

Setiap jenis peraturan punya subfolder sendiri dengan dua subdirektori, `metadata/` (JSON per dokumen) dan `pdfs/`. Registry gabungan ditulis di root `data/raw/`.

```
data/raw/
├── UU/
│   ├── metadata/
│   │   ├── uu-1-2026__337869.json
│   │   └── ...
│   └── pdfs/
│       ├── uu-1-2026.pdf
│       ├── uu-1-2026_lampiran_1.pdf
│       └── ...
├── PP/
│   ├── metadata/
│   └── pdfs/
├── PERPRES/
├── PERMEN_ESDM/
├── PERATURAN_BPOM/
├── ...
└── registry.json
```

### Format Nama File

**Metadata,** `{doc_id}__{detail_id}.json`. `doc_id` adalah `{bentuk_singkat}-{nomor}-{tahun}` (misal `uu-1-2026`), `detail_id` adalah ID unik dari situs BPK untuk hindari duplikasi.

**PDF,** semua deterministic by `doc_id`,
- PDF utama, `{doc_id}.pdf`
- Lampiran ke-N, `{doc_id}_lampiran_N.pdf`
- File tambahan ke-N (non-utama, non-lampiran), `{doc_id}_extra_N.pdf`

### Contoh Isi Metadata JSON

```json
{
  "detail_id": "337869",
  "slug": "uu-no-1-tahun-2026",
  "url": "https://peraturan.bpk.go.id/Details/337869/uu-no-1-tahun-2026",
  "materi_pokok": "UU ini mengatur mengenai ...",
  "judul": "Undang-undang (UU) Nomor 1 Tahun 2026 tentang Penyesuaian Pidana",
  "kategori": "Pusat",
  "nomor": "1",
  "bentuk": "Undang-undang (UU)",
  "bentuk_singkat": "UU",
  "tahun": "2026",
  "tempat_penetapan": "Jakarta",
  "tanggal_penetapan": "02 Januari 2026",
  "status": "Berlaku",
  "subjek": "HUKUM PIDANA, PERDATA, DAN DAGANG",
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
      "href": "/Download/400929/UU%20Nomor%201%20Tahun%202026.pdf",
      "local_path": "UU/pdfs/uu-1-2026.pdf"
    }
  ],
  "doc_id": "uu-1-2026",
  "pdf_path": "UU/pdfs/uu-1-2026.pdf",
  "lampiran_paths": ["UU/pdfs/uu-1-2026_lampiran_1.pdf"],
  "extra_paths": []
}
```

`pdf_path`, `lampiran_paths`, dan `extra_paths` ditulis setelah PDF berhasil di-download. Konsumen downstream sebaiknya pakai field ini untuk resolve path, bukan menebak dari `filename` asli.

### Registry (`registry.json`)

Satu file gabungan berisi ringkasan seluruh dokumen dari semua jenis. Di-generate otomatis di akhir proses dari semua subfolder `*/metadata/`.

```json
{
  "uu-1-2026": {
    "doc_id": "uu-1-2026",
    "detail_id": "337869",
    "jenis_folder": "UU",
    "kategori": "Pusat",
    "bentuk_singkat": "UU",
    "nomor": "1",
    "tahun": "2026",
    "judul": "Undang-undang (UU) Nomor 1 Tahun 2026 tentang Penyesuaian Pidana",
    "status": "Berlaku",
    "tanggal_penetapan": "02 Januari 2026",
    "relasi": [
      {"tipe": "Mengubah", "ref": "UU No. 1 Tahun 2023", "ref_id": "234935"}
    ],
    "has_pdf": true,
    "pdf_path": "UU/pdfs/uu-1-2026.pdf",
    "lampiran_paths": ["UU/pdfs/uu-1-2026_lampiran_1.pdf"]
  }
}
```

Registry merge-aware. Jalankan ulang scraper akan menggabungkan entry baru ke registry yang sudah ada, bukan overwrite.

## Fitur

- **Resume.** `--resume` skip dokumen yang sudah ada berdasarkan `detail_id`.
- **Auto-skip dropped docs.** Saat startup, scraper baca `data/dropped_docs.json` dan otomatis tambahkan semua doc_id di sana ke skip set. Doc yang pernah di-drop karena MAJOR/FAIL/ERROR ga akan re-fetched ulang waktu expand kategori. Lihat `scripts/parser/corpus_status.py`.
- **Retry.** Setiap request otomatis di-retry hingga 3 kali pada error transien.
- **Short-circuit pada status non-recoverable.** Status 400, 401, 403, 404, 405, 410, 451 tidak di-retry, langsung return None agar tidak buang waktu.
- **Streaming PDF.** PDF di-download streaming agar tidak membebani memori.
- **Polite scraping.** Default delay 1.5 detik antar-request.

## Catatan

- Situs BPK pakai Cloudflare. Scraper mengandalkan `requests.Session` untuk pertahankan cookies dari halaman pencarian agar bisa akses halaman detail.
- Kalau scraper mulai dapat error 403, coba naikkan `--delay`.
- 10 dokumen per halaman pencarian.
