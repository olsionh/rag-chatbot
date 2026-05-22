from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Load PDF
print("Loading document...")
loader = PyPDFLoader("projektligji.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# Step 3: Create vector store
print("Creating embeddings... (takes ~30 seconds)")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Step 4: Build QA chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini")
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 5: Ask questions in a loop
print("\nReady! Ask questions about your document. Type 'quit' to exit.\n")
while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
