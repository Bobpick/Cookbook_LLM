# Voice-Enabled Recipe RAG Assistant

A voice- or text-based AI assistant for querying recipes from your cookbook collection. It uses Retrieval-Augmented Generation (RAG) to search a vectorized corpus of PDFs, then generates natural, US-measurement recipes (or lists) via Ollama. Supports voice input/output for hands-free kitchen use!

************************************************
THIS IS VERY IMPORTANT - IT IS NOT A FAST CODE
Files need to download on the first run. When you query the RAG it may take a minute to give you the material you are looking for!!
************************************************
![Voice-Enabled Recipe RAG Assistant Demo](cookbook.jpg)
## Features

- **PDF Ingestion**: Extracts text from cookbooks (with OCR fallback for scanned pages) using PyMuPDF + Tesseract. Chunks content for precise retrieval.
- **Semantic Search**: Uses Sentence Transformers to embed and retrieve relevant passages.
- **Dynamic Generation**:
  - List queries (e.g., "list of pork recipes") → Clean, numbered roundups.
  - Single recipes (e.g., "recipe for pork chops") → Structured format (Ingredients, Instructions, Tips).
- **Voice Mode**: Record queries via microphone, get spoken responses (uses your `voice_agent.py`).
- **US Measurements**: All outputs adapted to cups, tsp, °F, etc.
- **Incremental Updates**: Only re-process new/updated PDFs.
- **Fallbacks**: Keyword search if semantic matches are weak; handles sparse context gracefully.

Built for home cooks – query like "easy chicken dinners" and get grounded, chatty responses!

## Demo

```
You said: please provide a list of pork recipes

Recipe List:
Pork lovers, rejoice! Here's a quick roundup from the healthy dinners book—all simple and flavorful.

1. Pork Mignons with French Applesauce: Tender pork tenderloin medallions (4 oz each) pan-seared and topped with applesauce from 2 peeled apples, cinnamon, and lemon juice. Cook 10 mins total—serves 4.

2. Pork Chops in Warm Cherry Sauce: 1-inch thick chops browned in a skillet, then simmered with 1 cup cherries and balsamic glaze for 15 mins. Sweet-savory balance for 4.

Which one sounds good?
```

## Installation

### Clone the Repo

```bash
git clone https://github.com/BobPick/Cookbook_LLM.git
cd voice-recipe-rag
```

### Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

Create `requirements.txt` with the following content:

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
ollama>=0.1.0
numpy>=1.24.0
PyMuPDF>=1.23.0  # For PDF extraction
PyPDF2>=3.0.0    # Fallback (optional)
pillow>=10.0.0   # For OCR images
pytesseract>=0.3.10
tqdm>=4.65.0
textwrap  # Built-in, but for clarity
```

Then install:

```bash
pip install -r requirements.txt
```

### Install Tesseract OCR (for scanned PDFs)

- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

### Download Models (first run auto-downloads)

- **Ollama**: Run `ollama pull llama3.1` (or your preferred model).
- **Hugging Face**: BART-large-CNN (summarization, ~1.6GB) and paraphrase-MiniLM-L12-v2 (~80MB) – downloaded on first ingest.

### Configure Paths

Edit `ingest_pdfs.py` and `rag_query.py`:

- `pdf_directory`: Path to your cookbook PDFs (e.g., `/home/user/Documents/PDFs/cookbooks`).
- `model_directory`: Output for corpus/embeddings (e.g., `/home/user/Documents/PATL/model/cookbooks`).

## Usage

### Build the Corpus (run once, or when adding PDFs)

```bash
python ingest_pdfs.py
```

- Processes PDFs in parallel (with OCR fallback).
- Chunks text (~700 chars each) and embeds for RAG.
- Saves to `model_directory` (`corpus.pkl`, `embeddings.npy`, `titles.pkl`).
- Incremental: Only re-processes new/updated files.

### Query Recipes (voice or text)

```bash
python rag_query.py
```

- Choose `[v]` for voice or `[t]` for text.
- Examples:
  - Voice: "List of chicken recipes" → Numbered list.
  - Text: "Recipe for pork chops in cherry sauce" → Full structured recipe.
- Outputs printed/wrapped (console) and spoken (unwrapped for natural flow).
- Type/say "quit" to exit.

## Project Structure

- `rag_query.py`: Core RAG system (retrieval + Ollama generation). Handles voice, lists/single recipes, US units.
- `ingest_pdfs.py`: PDF-to-corpus pipeline (extraction, chunking, embedding).
- `voice_agent.py`: Your audio utilities (record, STT, TTS) – add if not present.
- `model/`: Generated data (add to `.gitignore`).
- `.gitignore`: Ignore `model/`, `__pycache__`, `.venv/`.

## Customization

- **Ollama Model**: Swap `"llama3.1"` in `rag_query.py` for others (e.g., `"gemma:2b"` for lighter).
- **Chunk Size**: Adjust `chunk_size=700` in `ingest_pdfs.py` for finer/coarser retrieval.
- **Voice Duration**: Change `duration=6` in recording for longer queries.
- **Thresholds**: Tweak `low_threshold=0.25` in retrieval for stricter keyword fallbacks.

## Limitations & Notes

- **Corpus Size**: Caps at ~350k chunks to avoid overload; scale down `k=10` in retrieval if needed.
- **Hallucinations**: Grounded in context, but sparse PDFs (e.g., just titles) yield high-level summaries.
- **Performance**: First ingest downloads models (~2GB); use GPU for faster embedding if available.
- **Voice**: Requires `voice_agent.py` (not included – implement with libraries like `speech_recognition`, `pyttsx3`, or ElevenLabs).
- **Errors**: TensorFlow warnings are benign (from Sentence Transformers); suppress with `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`.

## Contributing

Fork, PRs welcome! Issues for bugs/features (e.g., more ingredients in regex, metric support toggle).

## License

MIT – See [LICENSE](LICENSE) (add one if needed).
