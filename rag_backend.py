import os
import urllib.request
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse


# ---- Config (edit if you want) ----
PDF_URL = os.getenv(
    "HR_PDF_URL",
    "https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf",
)
PDF_PATH = os.getenv("HR_PDF_PATH", "Leave-Policy-India.pdf")

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")

# Claude v2 is EOL in Bedrock. Use a Converse-supported model id.
# This "us." inference profile routes to us-east-1 / us-west-2 as per AWS Converse docs.
LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
)

# Titan embeddings model id (v1 works for many accounts; if yours differs, update here)
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))


def _download_pdf(url: str, path: str) -> str:
    """Download the PDF locally so PyPDFLoader can read it."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def hr_index() -> FAISS:
    """
    Builds and returns a FAISS vector DB (kept name as hr_index so your frontend can call it).
    """
    pdf_path = _download_pdf(PDF_URL, PDF_PATH)
    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embeddings = BedrockEmbeddings(
        region_name=AWS_REGION,
        credentials_profile_name=AWS_PROFILE,
        model_id=EMBED_MODEL_ID,
    )

    db = FAISS.from_documents(chunks, embeddings)
    return db


def _hr_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        region_name=AWS_REGION,
        credentials_profile_name=AWS_PROFILE,
        model_id=LLM_MODEL_ID,
        temperature=0.1,
        max_tokens=800,
    )


def _format_context(docs) -> str:
    # Keep it simple: join raw text
    return "\n\n".join(d.page_content for d in docs)


def hr_rag_response(index: FAISS, question: str) -> str:
    """
    Takes FAISS db as `index` (kept param name to match your frontend),
    retrieves context, and asks Bedrock chat model.
    """
    question = (question or "").strip()
    if not question:
        return "Ask a non-empty question."

    # Retrieve
    docs = index.similarity_search(question, k=TOP_K)
    context = _format_context(docs)

    # Ask LLM (Converse-style messages)
    llm = _hr_llm()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "You are an HR policy assistant.\n"
                        "Answer the question using ONLY the context. "
                        "If the answer is not in the context, say you don't have enough information.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
        }
    ]

    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as e:
        # Common causes:
        # - model access not enabled
        # - wrong region/model_id
        return f"Bedrock error: {e}"
