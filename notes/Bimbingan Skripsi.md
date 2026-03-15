# Bimbingan Skripsi

## Ide Awal
*Date: 30 January 2026*

"Hierarchical Navigation of Indonesian Regulatory Frameworks Using a Vectorless RAG for Multi-Level Legal Question Answering"

Description
This research develops a reasoning-based QA system that replaces vector similarity with a Structural Tree Index of Indonesian statutes. The system will index the hierarchy of Indonesian regulations (UU, PP, Perpres) according to UU No. 12 Tahun 2011. When a user asks a general legal question (e.g., "What are the requirements for a digital signature?"), the agent doesn't just look for "keywords"; it navigates the Table of Contents (ToC) of relevant laws to find the specific Bab (Chapter), Pasal (Article), and Ayat (Paragraph) that govern the topic, ensuring the answer is grounded in the correct legislative level.

(possible) Research Questions
- How does a reasoning-based tree traversal compare to Top-K vector retrieval in identifying the correct Pasal (Article) for cross-regulatory legal queries?
- To what extent can a vectorless PageIndex approach resolve contradictions between a higher-level Law (UU) and its implementing Regulation (PP)?
- How does the "Reasoning Path" (traceability) provided by PageIndex improve user trust in AI-generated legal advice compared to "black-box" vector RAG?

Reference: Vectify AI (2024): PageIndex: Vectorless, Reasoning-Based RAG.

## Bimbingan 1
*Date: 11 February 2026*

- Sumber PDF dari web BPK 
- Jangan ambil semua sekaligus dulu, tapi sample dulu biar semua kategori ada 
- Better kalau pusat, kementerian, daerah juga ada 
- Cek apakah sisi retrieval antara vector RAG dengan vectorless RAG bisa dicompare atau gak, kalaubisa, compare buat di RQ 
- TODO: Dataset buat comparison?? Bisa tanya dari Joel 
- Di PageIndex asli, JSON cumin pointer ke halaman PDF berdasarkan ToC, di skripsi, better tulis isi per pasal juga 
- Fokus ke backend, frontend ambil dari Lexin 
- Pake Gemini API buat prompt engineeringnya 
- Monograf easier, jurnal lebih tricky 
- Bebas Indo or English as long proper tulisannya 
- Use AI tools wisely

## Bimbingan 2
*Date: 18 February 2026*

- Butuh dataset retrieval, input question, answer chunk (bisa create pake LLM) 
- Database di vector rag sama vectorless rag harus sama 
- Memotong di vector RAG (contoh: Lexin) bisa berbagai cara (per pasal, per halaman, dll) 
- Vectify RAG - Cari paper atau research terkait metrik

## Bimbingan 3
*Date: 25 February 2026*

- Eksperimen hasil kalau ayat displit vs tidak displit 
- Eksperimen hasil kalau retrieval llm only, bm25 only, hybrid 
- Buat setiap dokumen index berarti ada 2 jenis (ayat displit vs tidak displit)

## Bimbingan 4
*Date: 04 March 2026*

- Eksperimen buat RQ1: retrieval jenis bm25-llm-hybrid, leaf di pasal-ayat-dalem, best resultnya dicompare ke vector RAG 
- RQ2 dibenerin, RQ3 ga usah, RQ1 juga benerin wordingnya 
- BM25 ga perlu experiment parameter, nanti fokusnya malah ke BM25 
- Referensi parameter BM25 
- LLM gapapa buat rerank
- Current judul (EN): Hierarchical Navigation of Indonesian Regulatory Frameworks Using a Vectorless RAG for Multi-Level Legal Question Answering dan ID nya Navigasi Hierarkis Kerangka Regulasi Indonesia Menggunakan RAG Tanpa Vektor untuk Sistem Tanya Jawab Hukum Multi-Level

## Bimbingan 5
*Date: 11 March 2026*

- Revisi RQ:
    (1) How do leaf node granularity levels (Article, Paragraph, and Sub-paragraph) and retrieval methods (BM25, Full LLM, and Hybrid) affect retrieval accuracy in a Vectorless RAG system for Indonesian regulatory documents?
    (2) How does the optimal Vectorless RAG configuration compare to Vector RAG in retrieval accuracy for Indonesian regulatory documents?
- Kepikiran nambah RQ (3) To what extent does query formulation style, specifically formal legal language versus colloquial natural language, affect the retrieval accuracy of Vectorless RAG configurations relative to Vector RAG? tapi Pak Adila suggest fokus ke RQ 1 dan RQ 2 dulu aja takutnya terlalu luas, kalau cepet beres baru boleh nambah skenario.
- Di experiment perbandingan dengan vector RAG, buat beberapa skenario dulu (misal: per pdf, per halaman), jangan langsung ambil 1 setup. Jadi both vectorless dan vector RAG ada skenario.
- Saya concern judulnya "RAG" tapi RQ nya fokus retrieval, kata Pak Adila gapapa, nanti dijelasin aja, lagian vectorless itu terobosan barunya di sisi retrieval, generationnya sama aja. Buat judul bisa revisi selama pengerjaan.