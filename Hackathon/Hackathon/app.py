import os
import requests
import gradio as gr
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer

ENDPOINT_NAME = "ipo_forensics_endpoint"
INDEX_NAME = "workspace.default.ipo_chunks_v3_index"
SARVAM_API_KEY = "your_api_key"
COMPANIES = ["Aequs", "Aether", "BlueStone", "CP_Plus", "Groww", "ICICI_AMC","Pine_labs"]

vsc = VectorSearchClient(
    workspace_url=os.getenv("DATABRICKS_HOST"),
    personal_access_token=os.getenv("DATABRICKS_TOKEN")
)
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
            "max_tokens": 1024
        }
    )
    data = response.json()
    if "choices" not in data:
        raise ValueError(f"Sarvam API error: {data}")
    return data["choices"][0]["message"]["content"]

def get_forensics_report(company):
    if not company:
        return "please select a company"
    try:
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
            ctx = get_context(company, query, doc_type=doc_type, k=2)  
            all_context.append(f"### {query.upper()}\n{ctx}")
        full_context = "\n\n".join(all_context)
        prompt = FORENSICS_TEMPLATE.format(company=company, context=full_context)
        return call_llm(prompt)
    except Exception as e:
        return f"ERROR: {str(e)}"

def ask_question(company, question):
    if not company:
        return "please select a company"
    if not question.strip():
        return "please enter a question"
    try:
        context = get_context(company, question, k=10)
        prompt = QA_TEMPLATE.format(company=company, question=question, context=context)
        return call_llm(prompt)
    except Exception as e:
        return f"ERROR: {str(e)}"

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
    demo.launch(server_name="0.0.0.0", server_port=8000)