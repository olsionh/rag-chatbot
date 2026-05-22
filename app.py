import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
import os

# Page config
st.set_page_config(page_title="RAG Document Chat", page_icon="📄")
st.title("📄 Chat with your Documents")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_files = st.file_uploader(
        "Upload PDFs (multiple allowed)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and api_key:
        if st.button("Index Documents"):
            with st.spinner("Reading and indexing all documents..."):
                os.environ["OPENAI_API_KEY"] = api_key

                all_chunks = []

                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()

                    # Tag each chunk with filename
                    for page in pages:
                        page.metadata["source"] = uploaded_file.name

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = splitter.split_documents(pages)
                    all_chunks.extend(chunks)
                    os.unlink(tmp_path)

                # Embed all chunks together
                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma.from_documents(all_chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                # Build chain
                prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based on the context and conversation history below.
If you don't know the answer from the context, say "I don't find this in the documents."

Conversation history:
{chat_history}

Context from documents:
{context}

Question: {question}
""")
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

                def format_docs(docs):
                    return "\n\n".join(
                        f"[{doc.metadata.get('source', 'unknown')} - page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                        for doc in docs
                    )

                def format_history(msgs):
                    lines = []
                    for m in msgs:
                        if isinstance(m, HumanMessage):
                            lines.append(f"User: {m.content}")
                        elif isinstance(m, AIMessage):
                            lines.append(f"Assistant: {m.content}")
                    return "\n".join(lines)

                st.session_state.chain = {
                    "prompt": prompt,
                    "llm": llm,
                    "format_docs": format_docs,
                    "format_history": format_history
                }

                st.session_state.messages = []
                st.session_state.chat_history = []

            total = len(uploaded_files)
            total_chunks = len(all_chunks)
            st.success(f"Indexed {total} file(s) — {total_chunks} chunks total.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(src)

# Chat input
if question := st.chat_input("Ask something about your documents..."):
    if not st.session_state.chain or not st.session_state.retriever:
        st.warning("Please upload PDFs and click 'Index Documents' first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Retrieve + answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = st.session_state.chain
                retriever = st.session_state.retriever

                # Get relevant docs
                docs = retriever.invoke(question)
                context = chain["format_docs"](docs)
                history = chain["format_history"](st.session_state.chat_history)

                # Build and run chain
                runnable = (
                    chain["prompt"]
                    | chain["llm"]
                    | StrOutputParser()
                )

                answer = runnable.invoke({
                    "context": context,
                    "question": question,
                    "chat_history": history
                })

                # Update memory
                st.session_state.chat_history.append(HumanMessage(content=question))
                st.session_state.chat_history.append(AIMessage(content=answer))

                # Keep history to last 10 messages
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]

                # Extract sources
                sources = list(set(
                    f"{doc.metadata.get('source', 'unknown')} — page {doc.metadata.get('page', '?')}"
                    for doc in docs
                ))

            st.write(answer)
            with st.expander("Sources"):
                for src in sources:
                    st.caption(src)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
