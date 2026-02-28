import streamlit as st
import os
import time
from datetime import datetime

# --- FORCE LIGHT MODE ---
st.set_page_config(
    page_title="NewsNexus AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL LIGHT THEME OVERRIDE ---
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

.main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}
.sub-header {font-size: 1.5rem; color: #4B5563;}
.agent-box {
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    background-color: #f9f9f9;
}
.success-box {
    background-color: #D1FAE5;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #10B981;
}
</style>
""", unsafe_allow_html=True)

# --- Import Backend Logic ---
from ingestion import ingest_documents
from tools import get_llm_with_tools, lookup_policy_docs, web_search_stub
from agents import app as agent_app
from memory_store import MemoryStore
from langchain_core.messages import HumanMessage

# --- Paths ---
PROJECT_ROOT = r"D:\python-project\news-nexus"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_data" not in st.session_state:
    st.session_state.research_data = []
if "chart_data" not in st.session_state:
    st.session_state.chart_data = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{int(time.time())}"
if "current_step" not in st.session_state:
    st.session_state.current_step = "idle"
if "draft_content" not in st.session_state:
    st.session_state.draft_content = ""

# --- PDF Export ---
def export_as_pdf(html_content):
    from io import BytesIO
    from xhtml2pdf import pisa
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)
    if pisa_status.err:
        return None
    return pdf_buffer.getvalue()

# --- Sidebar ---
with st.sidebar:
    st.title("📰 NewsNexus Control")
    st.divider()

    st.subheader("📂 Knowledge Base")

    existing_pdfs = []
    if os.path.exists(DATA_PATH):
        existing_pdfs = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if existing_pdfs:
        with st.expander(f"Documents in Library ({len(existing_pdfs)})"):
            for pdf in existing_pdfs:
                st.write(f"📄 {pdf}")
    else:
        st.caption("No documents in library yet.")

    st.divider()

    uploaded_files = st.file_uploader("Add New Reports (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(DATA_PATH, uploaded_file.name)
            os.makedirs(DATA_PATH, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} files!")
        st.rerun()

    if st.button("🧠 Build/Update Vector Index"):
        if not existing_pdfs:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Processing Library..."):
                try:
                    pages, chunks = ingest_documents()
                    st.success(f"Processed {pages} pages into {chunks} chunks.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    db_ready = os.path.exists(DB_PATH) and os.listdir(DB_PATH)
    status_msg = "✅ Database: ACTIVE" if db_ready else "⚠️ Database: MISSING"
    st.info(f"System Status: {status_msg}")
    st.caption(f"Mode: {'Hybrid (PDF+Web)' if existing_pdfs else 'Web Search Only'}")
    st.caption("LLM: Llama 3.x + nomic-embed-text")

# --- Main UI ---
st.markdown('<div class="main-header">📰 NewsNexus: Corporate Intelligence Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous Multi-Agent System with Human-in-the-Loop</div>', unsafe_allow_html=True)
st.divider()

topic = st.text_input("Enter Research Topic:")

if st.button("🚀 Start Agents", disabled=st.session_state.current_step != "idle") and topic:

    db_exists = os.path.exists(DB_PATH) and os.listdir(DB_PATH)
    raw_pdfs_exist = os.path.exists(DATA_PATH) and any(f.endswith(".pdf") for f in os.listdir(DATA_PATH))

    if db_exists:
        st.success("📂 Using existing Knowledge Base...")
    elif raw_pdfs_exist:
        with st.spinner("🔍 Indexing library..."):
            pages, chunks = ingest_documents()
            st.success(f"Library Indexed! ({pages} pages, {chunks} chunks)")
    else:
        st.warning("🌐 No PDFs found. Web Search Only Mode.")

    st.session_state.current_step = "researching"
    st.session_state.messages = [HumanMessage(content=topic)]
    st.session_state.research_data = []

    mem_store = MemoryStore()
    past_memory = mem_store.check_memory(topic)
    if "WARNING" in past_memory:
        st.warning(past_memory)

# --- Agent Execution ---
if st.session_state.current_step == "researching":

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": st.session_state.messages, "research_data": [], "chart_data": []}

    for event in agent_app.stream(inputs, config):
        if "Researcher" in event:
            st.session_state.research_data = event["Researcher"].get("research_data", [])

        if "Analyst" in event:
            st.session_state.chart_data = event["Analyst"].get("chart_data", [])

        if "Writer" in event:
            st.session_state.draft_content = event["Writer"]["messages"][-1].content

    st.session_state.current_step = "reviewing"
    st.rerun()

# --- Review Stage ---
if st.session_state.current_step == "reviewing":

    st.subheader("📝 Draft Review")

    st.components.v1.html(
        f"""
        <div style="background-color:white; color:black; padding:20px;">
        {st.session_state.draft_content}
        </div>
        """,
        height=600,
        scrolling=True
    )

    feedback = st.text_input("Feedback (Leave empty to approve):")

    if st.button("Submit Decision"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        if feedback:
            agent_app.update_state(config, {"messages": [HumanMessage(content=feedback)]})
            for event in agent_app.stream(None, config):
                pass
            state = agent_app.get_state(config)
            st.session_state.draft_content = state.values['messages'][-1].content
            st.rerun()
        else:
            st.session_state.current_step = "finished"
            mem_store = MemoryStore()
            topic_key = st.session_state.messages[0].content
            mem_store.save_memory(topic_key, st.session_state.draft_content)
            st.rerun()

# --- Final Stage ---
if st.session_state.current_step == "finished":
    st.balloons()
    st.markdown('<div class="success-box">✅ Newsletter Approved & Archived!</div>', unsafe_allow_html=True)

    st.download_button(
        label="📄 Download HTML",
        data=st.session_state.draft_content,
        file_name=f"newsletter_{int(time.time())}.html",
        mime="text/html"
    )

    pdf_data = export_as_pdf(st.session_state.draft_content)
    if pdf_data:
        st.download_button(
            label="📁 Download PDF",
            data=pdf_data,
            file_name=f"newsletter_{int(time.time())}.pdf",
            mime="application/pdf"
        )

    if st.button("🔄 New Research"):
        st.session_state.current_step = "idle"
        st.rerun()