import streamlit as st
from utils import read_pdf, save_uploaded_files
from rag_engine import RAG

st.set_page_config(page_title="📚 AI Book Q&A")

st.title("📚 Ask AI Anything from Your Books")

uploaded_files = st.file_uploader("Upload your books (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    file_paths = save_uploaded_files(uploaded_files)
    texts = [read_pdf(fp) for fp in file_paths]
    all_text = "\n".join(texts)

    st.success("Books uploaded and read!")

    if st.button("Process Books"):
        with st.spinner("Thinking..."):
            rag = RAG()
            chunks = rag.chunk_text(all_text)
            rag.build_index(chunks)
            st.session_state['rag'] = rag
        st.success("Books processed!")

if 'rag' in st.session_state:
    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Finding the best answer..."):
            answer = st.session_state['rag'].answer_question(query)
        st.write("### Answer:")
        st.write(answer)

        st.download_button(
            label="Download Answer",
            data=answer,
            file_name="answer.txt",
            mime="text/plain"
        )
