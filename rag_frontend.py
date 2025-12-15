import streamlit as st
import rag_backend as demo

st.set_page_config(page_title="HR Q and A with RAG", page_icon="🎯", layout="centered")

st.markdown(
    '<p style="font-family:sans-serif; color:Green; font-size: 42px;">HR Q & A with RAG 🎯</p>',
    unsafe_allow_html=True
)

st.caption("PDF → chunks → Titan embeddings → FAISS → retrieve → Claude (Bedrock Converse)")

@st.cache_resource
def get_vector_index():
    return demo.hr_index()

with st.spinner("📀 Building the vector database (first run only)..."):
    vector_index = get_vector_index()

input_text = st.text_area(
    "Input text",
    label_visibility="collapsed",
    placeholder="Ask something about the leave policy…",
    height=120,
)

go_button = st.button("Ask", type="primary")

if go_button:
    if not input_text.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Thinking..."):
            answer = demo.hr_rag_response(index=vector_index, question=input_text)
            st.write(answer)
