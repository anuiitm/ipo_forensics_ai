# Databricks notebook source
# MAGIC %pip install -U langchain sentence-transformers langchain-community langchain pypdf databricks-vectorsearch unstructured pdf2image pdfminer pillow typing-extensions>=4.12.0 --upgrade --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

base_path = "/Volumes/workspace/default/raw_data/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion and chunking

# COMMAND ----------

from pypdf import PdfReader
from langchain_core.documents import Document
import re

DOC_TYPE_MAP = {
    "concal.pdf": "concall_transcript",
    "drhp.pdf": "DRHP",
    "presentation.pdf": "quarterly_presentation"
}

def clean_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)      
    text = re.sub(r'\n+', '\n', text)     
    text = re.sub(r'[ \t]+', ' ', text)  
    return text.strip()

all_docs = []

for company in dbutils.fs.ls(base_path):
    company_name = company.name.replace("/", "")
    
    for file in dbutils.fs.ls(company.path):
        if not file.name.endswith(".pdf"):
            continue
            
        file_path = file.path.replace("dbfs:/", "/")
        doc_type = DOC_TYPE_MAP.get(file.name, file.name.replace(".pdf", ""))
        
        print(f"Processing: {file_path}")
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages):
            text = clean_text(page.extract_text() or "")
            if len(text) > 100:
                doc = Document(
                    page_content=text,
                    metadata={
                        "company": company_name,
                        "doc_type": doc_type,
                        "source": file.name,
                        "page": page_num + 1
                    }
                )
                all_docs.append(doc)
print(f"\n Total pages loaded: {len(all_docs)}")
print(f"   Companies: {sorted(set(d.metadata['company'] for d in all_docs))}")
print(f"   Doc types: {set(d.metadata['doc_type'] for d in all_docs)}")

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_CONFIG = {
    "DRHP": {
        "chunk_size": 1500,     
        "chunk_overlap": 300   
    },
    "concall_transcript": {
        "chunk_size": 800,      
        "chunk_overlap": 100
    },
    "quarterly_presentation": {
        "chunk_size": 500,      
        "chunk_overlap": 50
    }
}

DEFAULT_CONFIG = {"chunk_size": 1000, "chunk_overlap": 150}

def get_splitter(doc_type: str) -> RecursiveCharacterTextSplitter:
    config = CHUNK_CONFIG.get(doc_type, DEFAULT_CONFIG)
    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "], 
        length_function=len
    )

all_chunks = []

for doc in all_docs:
    splitter = get_splitter(doc["doc_type"] if isinstance(doc, dict) else doc.metadata["doc_type"])
    splits = splitter.split_documents([doc])
    for i, chunk in enumerate(splits):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(splits)
    
    all_chunks.extend(splits)

print(f"Total chunks created: {len(all_chunks)}")
from collections import Counter
breakdown = Counter(
    f"{c.metadata['company']} | {c.metadata['doc_type']}" 
    for c in all_chunks
)
print("\nChunk distribution:")
for key, count in sorted(breakdown.items()):
    print(f"  {key}: {count} chunks")

# COMMAND ----------

from pyspark.sql import Row
import json
rows = [
    Row(
        company=c.metadata["company"],
        doc_type=c.metadata["doc_type"],
        source=c.metadata["source"],
        page=c.metadata["page"],
        chunk_index=c.metadata["chunk_index"],
        total_chunks=c.metadata["total_chunks"],
        text=c.page_content
    )
    for c in all_chunks
]

df = spark.createDataFrame(rows)
df.write.mode("overwrite").saveAsTable("workspace.default.ipo_chunks")

print(f"Saved {df.count()} chunks to Delta table")

# COMMAND ----------

import json

chunks_serialized = [
    {
        "text": c.page_content,
        "metadata": c.metadata
    }
    for c in all_chunks
]

output_path = "/Volumes/workspace/default/raw_data/chunks.json"
with open(output_path, "w") as f:
    json.dump(chunks_serialized, f)

print(f"Saved {len(chunks_serialized)} chunks to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reloading the data from the delta table

# COMMAND ----------

df = spark.table("workspace.default.ipo_chunks")

from langchain_core.documents import Document

all_chunks = [
    Document(
        page_content=row["text"],
        metadata={
            "company": row["company"],
            "doc_type": row["doc_type"],
            "source": row["source"],
            "page": row["page"],
            "chunk_index": row["chunk_index"],
            "total_chunks": row["total_chunks"]
        }
    )
    for row in df.collect()
]

print(f"Reloaded {len(all_chunks)} chunks")

# COMMAND ----------

# with open("/Volumes/workspace/default/raw_data/chunks.json") as f:
#     chunks_serialized = json.load(f)

# all_chunks = [
#     Document(page_content=c["text"], metadata=c["metadata"])
#     for c in chunks_serialized
# ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embeddings and Vector Search

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df = spark.table("workspace.default.ipo_chunks") \
          .withColumn("chunk_id", monotonically_increasing_id().cast("string"))

df.write.mode("overwrite").saveAsTable("workspace.default.ipo_chunks_v2")

spark.sql("""
    ALTER TABLE workspace.default.ipo_chunks_v2
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

total = spark.table("workspace.default.ipo_chunks_v2").count()
nulls = spark.table("workspace.default.ipo_chunks_v2").filter("chunk_id IS NULL").count()
print(f"total rows: {total}, null chunk_ids: {nulls}")

# COMMAND ----------

df.limit(5).toPandas()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
ENDPOINT_NAME = "ipo_forensics_endpoint"

try:
    vsc.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"endpoint creation triggered")
    print("check status: Compute -> Vector Search in left sidebar")
except Exception as e:
    if "already exists" in str(e).lower():
        print("endpoint already exists")
    else:
        raise e

# COMMAND ----------

import time
def wait_for_endpoint(vsc, endpoint_name, timeout_mins=15):
    for i in range(timeout_mins * 2):
        state = vsc.get_endpoint(endpoint_name).get("endpoint_status", {}).get("state")
        if state == "ONLINE":
            print("endpoint is online")
            return
        print(f"[{i*30}s] current state: {state}")
        time.sleep(30)
    raise TimeoutError("endpoint did not come online in time")

wait_for_endpoint(vsc, ENDPOINT_NAME)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

INDEX_NAME = "workspace.default.ipo_chunks_v2_index"
SOURCE_TABLE = "workspace.default.ipo_chunks_v2"
EMBEDDING_MODEL = "databricks-bge-large-en"
ENDPOINT_NAME = "ipo_forensics_endpoint"

try:
    vsc.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text",
        embedding_model_endpoint_name=EMBEDDING_MODEL
    )
    print("index creation triggered")
except Exception as e:
    if "already exists" in str(e).lower():
        print("index already exists")
    else:
        raise e

# COMMAND ----------

def wait_for_index(vsc, endpoint_name, index_name, timeout_mins=30):
    for i in range(timeout_mins * 2):
        status = vsc.get_index(endpoint_name, index_name).describe().get("status", {})
        if status.get("ready"):
            print("index is ready")
            return
        print(f"[{i*30}s] {status.get('message', 'indexing...')}")
        time.sleep(30)
    raise TimeoutError("index did not become ready in time")

wait_for_index(vsc, ENDPOINT_NAME, INDEX_NAME)

# COMMAND ----------

index = vsc.get_index(ENDPOINT_NAME, INDEX_NAME)

results = index.similarity_search(
    query_text="objects of the issue IPO proceeds usage",
    columns=["chunk_id", "company", "doc_type", "text", "page"],
    num_results=3
)

for r in results.get("result", {}).get("data_array", []):
    print(f"company: {r[1]} | doc_type: {r[2]} | page: {r[4]}")
    print(f"preview: {r[3][:150]}")

# COMMAND ----------

from sentence_transformers import SentenceTransformer
from pyspark.sql.functions import col
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

df = spark.table("workspace.default.ipo_chunks_v2").toPandas()

print(f"embedding {len(df)} chunks...")
embeddings = model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)

df["embedding"] = embeddings.tolist()
print("done")

# COMMAND ----------

sdf = spark.createDataFrame(df)
sdf.write.mode("overwrite").saveAsTable("workspace.default.ipo_chunks_v3")
spark.sql("ALTER TABLE workspace.default.ipo_chunks_v3 SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"saved: {sdf.count()} rows")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()
try:
    index = vsc.create_delta_sync_index(
        endpoint_name="ipo_forensics_endpoint",
        index_name="workspace.default.ipo_chunks_v3_index",
        source_table_name="workspace.default.ipo_chunks_v3",
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_dimension=384,
        embedding_vector_column="embedding"
    )
    print("delta sync index created")
except Exception as e:
    if "already exists" in str(e).lower():
        print("index already exists")
        index = vsc.get_index("ipo_forensics_endpoint", "workspace.default.ipo_chunks_v3_index")
    else:
        raise e

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index("ipo_forensics_endpoint", "workspace.default.ipo_chunks_v3_index")

# Trigger sync to load your MiniLM embeddings
print("Triggering sync to load your MiniLM embeddings from workspace.default.ipo_chunks_v3...")
index.sync()
print("Sync triggered. Waiting for completion...\n")

for i in range(60):  # Wait up to 30 minutes
    desc = index.describe()
    num_rows = desc.get('num_indexed_rows', 0)
    status = desc.get('status', {})
    
    if status.get('ready') and num_rows > 0:
        print(f"\n✓ Index is ready!")
        print(f"  Indexed {num_rows:,} rows with your local MiniLM embeddings (384d)")
        break
    
    msg = status.get('message', f'Syncing... ({num_rows:,} rows so far)')
    print(f"[{i*30}s] {msg}")
    time.sleep(30)
else:
    print(f"\nSync in progress ({num_rows:,} rows so far). Check status at:")
    print("https://dbc-769a5f05-e643.cloud.databricks.com/explore/data/workspace/default/ipo_chunks_v3_index")

# COMMAND ----------

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
def get_context(company, query, doc_type=None, k=15):
    filters = {"company": company}
    if doc_type:
        filters["doc_type"] = doc_type
    query_embedding = embed_model.encode(query).tolist()
    results = index.similarity_search(
        query_vector=query_embedding,
        columns=["company", "doc_type", "text", "page"],
        filters=filters,
        num_results=k
    )
    rows = results.get("result", {}).get("data_array", [])
    if not rows:
        return "No relevant context found."
    parts = [f"[{r[1]} | Page {r[3]}]\n{r[2]}" for r in rows]
    return "\n\n---\n\n".join(parts)

# COMMAND ----------

# DBTITLE 1,Test vector search with MiniLM
test_query = "What are the objects of the IPO and how will the proceeds be used?"
test_company = "Groww" 
print(f"Query: {test_query}")
print(f"Company: {test_company}")
context = get_context(company=test_company, query=test_query, k=3)
print(context)

# COMMAND ----------

sk_kxgagvwn_hEHjKLxMUKp3ZjxuuxlRqV9M

# COMMAND ----------

import os
import requests
import gradio as gr
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer

ENDPOINT_NAME = "ipo_forensics_endpoint"
INDEX_NAME = "workspace.default.ipo_chunks_v3_index"
SARVAM_API_KEY = "sk_kxgagvwn_hEHjKLxMUKp3ZjxuuxlRqV9M"

COMPANIES = [
    "Aequs", "Aether", "BlueStone"
    # add your remaining companies exactly as stored in delta table
]

vsc = VectorSearchClient()
index = vsc.get_index(ENDPOINT_NAME, INDEX_NAME)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = """
You are a SEBI-registered research analyst and IPO forensics expert.
You analyze Indian IPO documents including DRHPs, quarterly presentations, and concall transcripts.

Rules:
- Only use information from the provided context
- Cite source as [DRHP], [Concall] or [Presentation] inline
- Flag red flags clearly with WARNING:
- If data is missing from context, say "Not available in provided documents"
- Be factual, structured, and concise
"""

FORENSICS_TEMPLATE = """
Generate a complete IPO forensics report for {company} using only the context below.

Cover all 10 sections:
1. Business Overview
2. Industry & Market Overview
3. Objects of the Issue
4. Financial Highlights (Revenue, EBITDA, PAT, Margins, ROE, ROCE, EPS)
5. Cash Flow Analysis
6. Risk Factors (top 5)
7. Promoter & Management Analysis
8. Related Party Transactions
9. Peer Comparison & Valuation
10. Red Flags

Context:
{context}
"""

QA_TEMPLATE = """
Answer the following question about {company} using only the context below.
If the answer is not in the context, say "Not available in provided documents".

Question: {question}

Context:
{context}
"""

def get_context(company, query, doc_type=None, k=15):
    filters = {"company": company}
    if doc_type:
        filters["doc_type"] = doc_type

    query_embedding = embed_model.encode(query).tolist()
    results = index.similarity_search(
        query_vector=query_embedding,
        columns=["company", "doc_type", "text", "page"],
        filters=filters,
        num_results=k
    )
    rows = results.get("result", {}).get("data_array", [])
    if not rows:
        return "No relevant context found."
    parts = [f"[{r[1]} | Page {r[3]}]\n{r[2]}" for r in rows]
    return "\n\n---\n\n".join(parts)

def call_llm(prompt):
    response = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "sarvam-m",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
    )
    data = response.json()
    if "choices" not in data:
        raise ValueError(f"Sarvam API error: {data}")
    return data["choices"][0]["message"]["content"]

def get_forensics_report(company):
    if not company:
        return "please select a company"
    sections = {
        "business products revenue model geographical presence": "DRHP",
        "objects of the issue IPO proceeds utilization": "DRHP",
        "risk factors key risks business threats": "DRHP",
        "promoter holding background management team": "DRHP",
        "related party transactions": "DRHP",
        "industry overview market size competition": "DRHP",
        "revenue EBITDA profit margins financial performance": "quarterly_presentation",
        "cash flow operations investing financing": "quarterly_presentation",
        "management commentary outlook guidance growth": "concall_transcript",
        "peer comparison valuation P/E EV EBITDA pricing": "DRHP"
    }
    all_context = []
    for query, doc_type in sections.items():
        ctx = get_context(company, query, doc_type=doc_type, k=5)
        all_context.append(f"### {query.upper()}\n{ctx}")
    full_context = "\n\n".join(all_context)
    prompt = FORENSICS_TEMPLATE.format(company=company, context=full_context)
    return call_llm(prompt)

def ask_question(company, question):
    if not company:
        return "please select a company"
    if not question.strip():
        return "please enter a question"
    context = get_context(company, question, k=10)
    prompt = QA_TEMPLATE.format(company=company, question=question, context=context)
    return call_llm(prompt)

with gr.Blocks(title="IPO Forensics AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IPO Forensics AI\nPowered by DRHP · Quarterly Reports · Concall Transcripts · Sarvam-m")

    with gr.Tab("Full Forensics Report"):
        company_select = gr.Dropdown(COMPANIES, label="Select Company")
        generate_btn = gr.Button("Generate Report", variant="primary")
        report_output = gr.Markdown()
        generate_btn.click(get_forensics_report, inputs=company_select, outputs=report_output)

    with gr.Tab("Ask Anything"):
        company_qa = gr.Dropdown(COMPANIES, label="Select Company")
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="What are the red flags in this IPO?"
        )
        ask_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Markdown()
        ask_btn.click(ask_question, inputs=[company_qa, question_input], outputs=answer_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)