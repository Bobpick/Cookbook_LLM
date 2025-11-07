#!/usr/bin/env python3
"""
PDF Ingestion for Recipe RAG Corpus.
Extracts text from cookbook PDFs (with OCR fallback), chunks it, embeds with Sentence Transformers,
and saves incrementally to disk for the RAG system.
"""
import os
import pickle
import re
import fitz  # pymupdf
from PyPDF2 import PdfReader  # Fallback for some PDFs (unused in code but imported for completeness)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import pytesseract
import time  # For mtime checks

# Optional: Suppress TF warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the tokenizer and model
print("Starting tokenizer download/load...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
print("Tokenizer ready.")

print("Starting model download/load (this may take 5-30+ min on first run)...")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
print("Model ready. Proceeding to embeddings...")

# Load the sentence transformer for embedding retrieval
print("Loading sentence embedder...")
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')
print("Embedder ready.")


def clean_extracted_text(text):
    """Clean fragmented PDF text (join mid-sentence breaks, remove extra spaces)."""
    # Fix fragmented sentences (e.g., "these. Days" -> "these days")
    text = re.sub(r'([a-zA-Z])(\.\s+)([a-zA-Z])', r'\1. \3', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean double newlines
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Join sentences
    return text.strip()


def extract_single(args):
    """Worker function for parallel PDF extraction with OCR fallback."""
    directory, filename, title, idx, total_pdfs = args
    filepath = os.path.join(directory, filename)
    try:
        # Prefer pymupdf for better handling
        doc = fitz.open(filepath)
        raw_text = ""
        ocr_used = False
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Try text layer first
            text = page.get_text().strip()
            if len(text) > 10:
                raw_text += text + "\n"
            else:  # OCR fallback for scanned pages
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, config='--psm 6')  # PSM 6 for block text (recipes)
                raw_text += ocr_text + "\n"
                ocr_used = True
        doc.close()

        if len(raw_text) > 10_000_000:  # Skip ultra-large pre-clean
            print(f"Skipping '{filename}': Too large ({len(raw_text)} chars)")
            return None, filename, title

        text = clean_extracted_text(raw_text)
        ocr_note = " (OCR used)" if ocr_used else ""
        if len(text) > 50:
            print(f"Working on PDF '{title}'{ocr_note}, number {idx} of {total_pdfs} (parallel)")
            return text, None, title
        else:
            print(f"Skipping '{filename}': Short extraction ({len(text)} chars)")
            return None, filename, title
    except Exception as e:
        print(f"Error reading PDF file: {filename} - {str(e)}")
        return None, filename, title


def read_pdf_files(directory, use_parallel=False, existing_titles=None, rag_mtime=None):
    """Read PDF files from a directory and extract the text content incrementally.
    - existing_titles: Set of existing titles to check for new PDFs.
    - rag_mtime: Timestamp of last RAG save; PDFs with later mtime are considered updated.
    Returns: new_corpus, new_titles, skipped_files
    """
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    total_pdfs = len(pdf_files)

    if total_pdfs == 0:
        print("No PDF files found in the directory.")
        return [], [], []

    # Filter to only new/updated PDFs
    new_pdf_files = []
    existing_titles_set = set(existing_titles) if existing_titles else set()
    for filename in pdf_files:
        title = os.path.splitext(filename)[0]
        filepath = os.path.join(directory, filename)
        file_mtime = os.path.getmtime(filepath)
        is_new = title not in existing_titles_set
        is_updated = rag_mtime is not None and file_mtime > rag_mtime
        if is_new or is_updated:
            new_pdf_files.append(filename)

    if not new_pdf_files:
        print("No new or updated PDFs found.")
        return [], [], []

    new_corpus = []
    new_titles = []
    skipped_files = []

    total_new = len(new_pdf_files)
    print(f"Processing {total_new} new/updated PDFs...")

    if use_parallel:
        args_list = [(directory, filename, os.path.splitext(filename)[0], idx + 1, total_new)
                     for idx, filename in enumerate(new_pdf_files)]

        with Pool(processes=4) as pool:  # Limit to 4 for OCR RAM safety
            results = list(tqdm(pool.imap(extract_single, args_list), total=total_new, desc="Extracting new PDFs"))

        for text, err_file, title in results:
            if err_file:
                skipped_files.append(err_file)
            else:
                new_corpus.append(text)
                new_titles.append(title)
    else:
        for idx, filename in enumerate(tqdm(new_pdf_files, desc="Extracting new PDFs"), start=1):
            title = os.path.splitext(filename)[0]
            print(f"Working on PDF '{title}', number {idx} of {total_new}")
            filepath = os.path.join(directory, filename)
            try:
                doc = fitz.open(filepath)
                raw_text = ""
                ocr_used = False
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text().strip()
                    if len(text) > 10:
                        raw_text += text + "\n"
                    else:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                        raw_text += ocr_text + "\n"
                        ocr_used = True
                doc.close()

                if len(raw_text) > 10_000_000:
                    print(f"Skipping '{filename}': Too large ({len(raw_text)} chars)")
                    skipped_files.append(filename)
                    continue

                text = clean_extracted_text(raw_text)
                ocr_note = " (OCR used)" if ocr_used else ""
                if len(text) > 50:
                    print(f"Working on PDF '{title}'{ocr_note}, number {idx} of {total_new}")
                    new_corpus.append(text)
                    new_titles.append(title)
                else:
                    print(f"Skipping '{filename}': Short extraction ({len(text)} chars)")
                    skipped_files.append(filename)
            except Exception as e:
                print(f"Error reading PDF file: {filename}")
                print(f"Error message: {str(e)}")
                skipped_files.append(filename)

    total_chars = sum(len(text) for text in new_corpus)
    print(
        f"New PDF extraction complete: {len(new_corpus)} new docs processed ({total_chars:,} total characters). Skipped: {len(skipped_files)} files")
    return new_corpus, new_titles, skipped_files


def encode_query(query):
    """Encode a query string into a vector representation."""
    query_embedding = embedder.encode(query)
    return query_embedding


# Example usage
pdf_directory = "/home/bob/Documents/PDFs/cookbooks"
use_parallel_extraction = True
enable_chunking = True  # Enabled for precision

model_directory = "/home/bob/Documents/PATL/model/cookbooks"
os.makedirs(model_directory, exist_ok=True)

# Load existing RAG data if available
corpus = []
titles = []
corpus_embeddings = None
rag_mtime = None
corpus_path = os.path.join(model_directory, 'corpus.pkl')
embeddings_path = os.path.join(model_directory, 'corpus_embeddings.npy')
titles_path = os.path.join(model_directory, 'titles.pkl')

if os.path.exists(corpus_path):
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    if os.path.exists(embeddings_path):
        corpus_embeddings = np.load(embeddings_path)
    if os.path.exists(titles_path):
        with open(titles_path, 'rb') as f:
            titles = pickle.load(f)
    rag_mtime = os.path.getmtime(corpus_path)
    print(f"Loaded existing RAG: {len(corpus)} chunks from {len(set(t for t in titles if '_' not in t))} original PDFs.")
else:
    print("No existing RAG found. Building from scratch.")

print("Starting PDF processing...")
new_corpus, new_titles = [], []  # Defaults if no new
if os.path.exists(pdf_directory):
    new_corpus, new_titles, _ = read_pdf_files(
        pdf_directory, use_parallel_extraction, existing_titles=[t.split('_chunk')[0] for t in titles], rag_mtime=rag_mtime
    )
else:
    print(f"PDF directory '{pdf_directory}' does not exist. Skipping extraction.")

if not new_corpus and not corpus:
    print("No corpus available (no new PDFs and no existing). Exiting.")
else:
    # Chunking with progress and size filter (only for new corpus)
    def chunk_text(text, chunk_size=700):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sent in sentences:
            test_chunk = current + sent + " "
            if len(test_chunk.encode('utf-8')) < chunk_size:
                current = test_chunk
            else:
                if len(current.strip()) > 100:  # Filter short chunks
                    chunks.append(current.strip())
                current = sent + " "
        if len(current.strip()) > 100:
            chunks.append(current.strip())
        return chunks

    if enable_chunking and new_corpus:
        print("Chunking new PDFs...")
        chunked_new_corpus = []
        chunked_new_titles = []
        total_chunks = len(corpus)  # Start from existing count
        for i, (text, title) in enumerate(tqdm(zip(new_corpus, new_titles), total=len(new_corpus), desc="Chunking new docs")):
            chunks = chunk_text(text)
            for j, chunk in enumerate(chunks):
                if total_chunks >= 350000:
                    print("Capped at 350k chunks to avoid overload.")
                    break
                chunked_new_corpus.append(chunk)
                chunked_new_titles.append(f"{title}_chunk{j}")
                total_chunks += 1
            if total_chunks >= 350000:
                break
        new_corpus = chunked_new_corpus
        new_titles = chunked_new_titles
        print(f"Chunked {len(new_corpus)} new passages.")

    # Append new to existing
    corpus.extend(new_corpus)
    titles.extend(new_titles)

    # Embed only new chunks
    if new_corpus:
        print("Encoding new corpus embeddings in batches...")
        batch_size = 20  # Smaller for stability
        new_embeddings = []
        for i in tqdm(range(0, len(new_corpus), batch_size), desc="Batching new embeddings"):
            batch = new_corpus[i:i + batch_size]
            batch_emb = embedder.encode(batch)
            new_embeddings.append(batch_emb)
            print(f"New batch {i // batch_size + 1}: {len(batch)} items embedded")
        new_embeddings = np.vstack(new_embeddings)

        # Append to existing embeddings
        if corpus_embeddings is not None:
            corpus_embeddings = np.vstack((corpus_embeddings, new_embeddings))
        else:
            corpus_embeddings = new_embeddings
        print("New embeddings ready.")

    print(f"Updated corpus: {len(corpus)} chunks from {len(set(t.split('_chunk')[0] for t in titles))} original PDFs.")

    # Save everything (overwrite)
    model.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)
    embedder.save(model_directory)
    print(f"Model saved to: {model_directory}")

    with open(corpus_path, 'wb') as f:
        pickle.dump(corpus, f)
    np.save(embeddings_path, corpus_embeddings)
    with open(titles_path, 'wb') as f:
        pickle.dump(titles, f)
    print("Updated corpus, embeddings, and titles saved.")
