# IPO Forensics AI

AI-powered forensics analysis of Indian IPOs using DRHP, quarterly presentations, and concall transcripts — built on Databricks.

---

## What It Does

IPO Forensics AI lets retail investors run institutional-grade analysis on any of the supported IPO companies. Select a company and get a structured 10-section forensics report covering business overview, financials, risk factors, promoter background, red flags, and more — all sourced directly from SEBI/BSE filings. Users can also ask freeform questions about any company in natural language.

---

## Architecture

```
BSE/SEBI PDFs (DRHP + Quarterly Presentation + Concall Transcript)
        │
        ▼
Databricks Volumes (raw PDF storage)
        │
        ▼
PySpark Ingestion (pypdf extraction, text cleaning, page-level chunking)
        │
        ▼
Delta Lake — ipo_chunks_v2 (persisted chunks with metadata)
        │
        ▼
Sentence Transformers (all-MiniLM-L6-v2, local embedding on cluster)
        │
        ▼
Databricks Vector Search — Direct Access Index (ipo_chunks_v3_index)
        │
        ▼
RAG Chain — company-filtered retrieval + section-specific queries
        │
        ▼
Sarvam-m (Indian LLM) via Sarvam AI API
        │
        ▼
Gradio UI — deployed as Databricks App
```

---

## Databricks Components Used

| Component | Usage |
|---|---|
| **Databricks Volumes** | Raw PDF storage under `/Volumes/workspace/default/raw_data/` |
| **Apache Spark / PySpark** | PDF ingestion, text extraction, parallel processing |
| **Delta Lake** | Persistent chunk storage with Change Data Feed enabled |
| **Databricks Vector Search** | Direct Access Index for semantic retrieval with company-level filters |
| **Databricks Apps** | Hosts the Gradio frontend as a deployed web application |
| **Unity Catalog** | Tables and index managed under `workspace.default` namespace |

---

## Supported Companies

- Aequs
- Aether
- BlueStone
- CP Plus
- Groww
- ICICI AMC
- Pine Labs

---

## How to Reproduce

### Prerequisites

- Databricks Free Trial workspace (not Community Edition)
- Sarvam AI API key — sign up free at [dashboard.sarvam.ai](https://dashboard.sarvam.ai)
- PDFs for each company: DRHP (from SEBI/BSE), quarterly investor presentation, and concall transcript (from BSE Announcements)

### Step 1 — Upload PDFs

Upload PDFs to Databricks Volumes with this structure:

```
/Volumes/workspace/default/raw_data/
├── Aequs/
│   ├── drhp.pdf
│   ├── concal.pdf
│   └── presentation.pdf
├── Aether/
│   ├── drhp.pdf
│   ├── concal.pdf
│   └── presentation.pdf
...
```

### Step 2 — Ingest and Chunk

Run `Finance_Forensics.py` (notebook) sequentially:

- Reads all PDFs via PySpark + pypdf
- Cleans text (fixes hyphenation, collapses whitespace)
- Splits into chunks (1500 chars for DRHP, 800 for concall, 500 for presentations)
- Saves to Delta table `workspace.default.ipo_chunks_v2` with Change Data Feed enabled

### Step 3 — Build Vector Index

In the same notebook:

```python
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# Create endpoint
vsc.create_endpoint(name="ipo_forensics_endpoint", endpoint_type="STANDARD")

# Compute embeddings locally (faster than managed embedding on free tier)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
df["embedding"] = model.encode(df["text"].tolist(), batch_size=64).tolist()

# Create Direct Access index with precomputed embeddings
vsc.create_direct_access_index(
    endpoint_name="ipo_forensics_endpoint",
    index_name="workspace.default.ipo_chunks_v3_index",
    primary_key="chunk_id",
    embedding_dimension=384,
    embedding_vector_column="embedding",
    schema={...}
)
```

### Step 4 — Deploy the App

Upload `app.py`, `app.yaml`, and `requirements.txt` to:
```
/Workspace/Users/<your-email>/Hackathon/
```

Update `app.yaml` with your credentials:
```yaml
command:
  - python
  - app.py
env:
  - name: DATABRICKS_HOST
    value: "https://<your-workspace>.cloud.databricks.com"
  - name: DATABRICKS_TOKEN
    value: "<your-personal-access-token>"
```

Run `deploy_app.py` to create and deploy the Databricks App. Once state shows `ACTIVE`, the app URL is live.

### Step 5 — Generate a Personal Access Token

Databricks UI → top right profile → Settings → Developer → Access Tokens → Generate New Token

---

## How to Use the App

**Full Forensics Report tab**
1. Select a company from the dropdown
2. Click "Generate Report"
3. Wait ~30 seconds for the 10-section report to appear
4. Click "Listen to Report" to hear the report via Sarvam TTS (Bulbul v3)

**Ask Anything tab**
1. Select a company
2. Type any question — e.g. "What are the red flags?", "Who are the promoters?", "What is the debt repayment plan?"
3. Click Ask — response is sourced only from that company's documents

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data Storage | Databricks Volumes + Delta Lake |
| Processing | PySpark + pypdf |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Store | Databricks Vector Search (Direct Access Index) |
| LLM | Sarvam-m via Sarvam AI API (Indian LLM) |
| TTS | Sarvam Bulbul v3 (Indian accent, English) |
| UI | Gradio |
| Hosting | Databricks Apps |

---

## Project Structure

```
Hackathon/
├── app.py                  # Gradio app — RAG chain, TTS, UI
├── app.yaml                # Databricks App config and env vars
├── requirements.txt        # Python dependencies
├── deploy_app.py           # Notebook to create and deploy the app
└── Finance_Forensics.py    # Ingestion, chunking, vector index notebook
```

---

## Notes

- The Vector Search index uses a Direct Access pattern with locally precomputed embeddings — this is significantly faster than managed embedding on Databricks Free Edition which throttles to ~1 row/second
- Sarvam-m context window is 7192 tokens — context per section is kept to 1-2 chunks to leave room for generation
- All data sourced from public SEBI/BSE filings — no proprietary data used
