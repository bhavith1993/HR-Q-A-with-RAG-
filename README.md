# HR Q&A with RAG 🎯 (Streamlit + Amazon Bedrock + FAISS)

A simple Retrieval-Augmented Generation (RAG) app that answers HR policy questions using:
- **Streamlit** UI
- **Amazon Bedrock** for:
  - embeddings (Amazon Titan)
  - chat completions (Claude via the **Converse** API)
- **FAISS** as an in-memory vector database
- **LangChain** integrations to glue everything together

> Note: Bedrock **Converse** provides a consistent chat interface across supported models, and Bedrock states it doesn’t store your prompts/documents as content. :contentReference[oaicite:0]{index=0}

---

## What this app does

1. Downloads an HR policy PDF locally
2. Loads and parses it into documents
3. Splits the content into chunks
4. Creates embeddings using **Bedrock Titan embeddings**
5. Stores embeddings in **FAISS**
6. On each question:
   - retrieves top-k matching chunks
   - sends **(context + question)** to a Bedrock chat model (Claude)
   - returns the answer

---

## Architecture

**User → Streamlit Frontend → Backend RAG Pipeline**
- **Frontend (Streamlit):** collects user question, shows response
- **Backend (Python):**
  - PDF download + load (PyPDFLoader)
  - chunking (RecursiveCharacterTextSplitter)
  - embeddings (BedrockEmbeddings)
  - vector store (FAISS)
  - LLM answer (ChatBedrockConverse)

**AWS Services Used**
- **Amazon Bedrock**
  - Embeddings model: `amazon.titan-embed-text-v1` (configurable)
  - Chat model: Claude (Converse-supported model id; configurable)
- **AWS IAM**
  - permissions to call Bedrock and read model access settings
- **Amazon CloudWatch**
  - metrics for Bedrock invocations can be viewed in CloudWatch metrics :contentReference[oaicite:1]{index=1}
  - optional model invocation logging can write invocation logs to CloudWatch Logs (and optionally S3) :contentReference[oaicite:2]{index=2}
- **AWS CloudTrail**
  - logs Bedrock API calls for auditing (including console + API calls) :contentReference[oaicite:3]{index=3}

---

## Repo structure

HR-Q&A-with-RAG/
├─ app.py # Streamlit frontend
├─ rag_backend.py # Backend: load -> chunk -> embed -> FAISS -> retrieve -> Bedrock Converse
├─ requirements.txt # (recommended) dependencies
└─ README.md

## Prerequisites

### 1) AWS account + Bedrock enabled


### 2) AWS credentials on your machine


### 3) Python 3.12+ recommended

## Installation

### Create & activate a virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

## Configuration 

AWS_REGION=us-west-2
AWS_PROFILE=default

# Models
EMBED_MODEL_ID=amazon.titan-embed-text-v1
LLM_MODEL_ID=us.anthropic.claude-3-5-sonnet-20240620-v1:0

# Document
HR_PDF_URL=https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf
HR_PDF_PATH=Leave-Policy-India.pdf

# Chunking / Retrieval
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=4

