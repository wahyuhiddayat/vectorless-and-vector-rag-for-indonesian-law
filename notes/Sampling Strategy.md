# Sampling Strategy — BPK JDIH Dataset

## Context

Dosbing minta semua kategori BPK ada di dataset (56 jenis dari Pusat + Daerah + Kementerian/Lembaga). Kategori besar di-sample, tidak diambil semua. Filter by tahun sebagai mekanisme sampling.

Survey sudah dijalankan: `scraper/data/survey_results.json` (2026-02-26).

## Survey Results Summary

| Group | Kategori | Total Docs | Biggest |
|-------|----------|------------|---------|
| Pusat | 8 | 17,277 | Keppres 6,968 / PP 4,971 |
| Daerah | 7 | 257,818 | Perbup 140K / Perda 62K |
| Kementerian | 41 | 11,006 | PMK 3,879 / Permenhub 1,341 |
| **Total** | **56** | **286,101** | |

## Full Strategy: 5 Tiers (untuk nanti)

Target: ~5,000-8,000 docs. Estimasi Gemini OCR cost: ~$30-50.

### Tier 1: Take all (no filter) — 55 categories, ~3,400 docs
Kategori dengan total <= 500. Langsung scrape semua.

### Tier 2: Filter 2020-2026 — 13 categories, ~1,850 docs
Kategori 200-500 total. Exception: BKN (217) dan Permenristekdikti (358) take all karena data mostly pre-2020.

| Jenis ID | Name | Total | 2020-2026 |
|----------|------|-------|-----------|
| 122 | Peraturan BKN | 217 | 0 → take all |
| 62 | Peraturan Bawaslu | 219 | 62 |
| 105 | Permenaker | 271 | 118 |
| 69 | Permenperin | 295 | 252 |
| 59 | Peraturan KPU | 297 | 86 |
| 111 | Permen ATR-BPN | 305 | 129 |
| 103 | Permensos | 336 | 73 |
| 110 | Permenristekdikti | 358 | 3 → take all |
| 13 | Inpres | 459 | 41 |
| 47 | Permenhan | 460 | 74 |
| 106 | Permenkominfo | 462 | 52 |
| 80 | Peraturan OJK | 477 | 169 |
| 46 | Permenkumham | 478 | 155 |

### Tier 3: Filter 2020-2026 — 7 categories, ~1,750 docs
Kategori 500-2,000 total.

| Jenis ID | Name | Total | 2020-2026 |
|----------|------|-------|-----------|
| 34 | Kanun | 522 | 174 |
| 78 | Peraturan BI | 631 | 92 |
| 107 | Permenpan-RB | 743 | 303 |
| 40 | Permendagri | 785 | 263 |
| 67 | Permendag | 991 | 340 |
| 48 | Permenhub | 1,341 | 327 |
| 8 | UU | 1,923 | 248 |

### Tier 4: Filter 2024-2026 — 4 categories, ~630 docs
Kategori 2,000-7,000 total.

| Jenis ID | Name | Total | 2024-2026 |
|----------|------|-------|-----------|
| 11 | Perpres | 2,640 | 278 |
| 42 | PMK | 3,879 | 209 |
| 10 | PP | 4,971 | 95 |
| 12 | Keppres | 6,968 | 48 |

### Tier 5: Filter 2025 + cap 200 — 4 categories, ~800 docs
Daerah besar (>10,000 total).

| Jenis ID | Name | Total | 2025 only | Cap 200 |
|----------|------|-------|-----------|---------|
| 20 | Pergub | 18,035 | 640 | 200 |
| 30 | Perwali | 36,808 | 1,356 | 200 |
| 19 | Perda | 62,038 | 1,301 | 200 |
| 23 | Perbup | 140,381 | 5,667 | 200 |

### Full Strategy Commands

```bash
# TIER 1: Take all
python scraper/bpk_scraper.py --jenis 273 77 219 90 88 86 28 76 118 38 --skip-pdf
python scraper/bpk_scraper.py --jenis 81 121 223 50 92 51 93 228 61 56 --skip-pdf
python scraper/bpk_scraper.py --jenis 123 99 35 58 100 101 89 27 83 66 --skip-pdf
python scraper/bpk_scraper.py --jenis 98 113 119 49 43 71 116 54 225 246 --skip-pdf
python scraper/bpk_scraper.py --jenis 95 114 124 109 255 87 52 108 104 53 --skip-pdf
python scraper/bpk_scraper.py --jenis 45 73 9 36 112 --skip-pdf

# TIER 2: 200-500, filter 2020-2026
python scraper/bpk_scraper.py --jenis 122 110 --skip-pdf  # take all (mostly old)
python scraper/bpk_scraper.py --jenis 62 105 69 59 111 103 13 47 106 80 46 --tahun 2020-2026 --skip-pdf

# TIER 3: 500-2000, filter 2020-2026
python scraper/bpk_scraper.py --jenis 34 78 107 40 67 48 8 --tahun 2020-2026 --skip-pdf

# TIER 4: 2000-7000, filter 2024-2026
python scraper/bpk_scraper.py --jenis 11 42 10 12 --tahun 2024-2026 --skip-pdf

# TIER 5: Daerah besar, filter 2025 + cap pages
python scraper/bpk_scraper.py --jenis 20 --tahun 2025 --pages 1-20 --skip-pdf
python scraper/bpk_scraper.py --jenis 30 --tahun 2025 --pages 1-20 --skip-pdf
python scraper/bpk_scraper.py --jenis 19 --tahun 2025 --pages 1-20 --skip-pdf
python scraper/bpk_scraper.py --jenis 23 --tahun 2025 --pages 1-20 --skip-pdf
```

## Quick Start: 10 per category (560 docs)

Ambil 10 docs (page 1) per kategori dulu. Biar semua 56 kategori ada representasi minimal.

```bash
# All 56 categories, page 1 only (10 docs each), metadata only
python scraper/bpk_scraper.py --jenis 8 9 10 11 12 13 36 273 19 20 23 27 28 30 34 35 38 40 42 43 45 46 47 48 49 50 52 53 54 56 58 59 61 62 66 67 69 71 73 75 76 77 78 80 81 83 86 87 88 89 90 92 93 95 98 99 100 101 103 104 105 106 107 108 109 110 111 112 113 114 116 118 119 121 122 123 124 219 223 225 228 246 255 --pages 1 --skip-pdf
```

Estimasi: ~83 list pages + ~560 detail pages = ~643 requests @ 1.5s = **~16 min**, **$0 cost**.

Setelah itu bisa scale up per tier.

## Cost Estimation (Gemini OCR)

| Scenario | Docs | Est. Pages | Gemini Cost |
|----------|------|------------|-------------|
| Quick start (10/cat) | 560 | ~5,600 | ~$3 |
| Full strategy | ~8,400 | ~82,000 | ~$30-50 |

Gemini 2.5 Flash pricing: ~$0.15/1M input, ~$0.60/1M output.
Per page: ~1,000 input + ~500 output tokens → ~$0.00045/page.
