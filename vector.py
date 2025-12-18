from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Load RTF correctly
loader = TextLoader("/Users/anushkagattani/Desktop/me.txt")
documents = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

docs = text_splitter.split_documents(documents)


from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Vector Store (FAISS) ---
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)

# Make a retriever for later RAG
retriever = vectorstore.as_retriever()

# Quick test: get 1 relevant chunk for a sample query
query = "Explain what this document is about"
results = retriever.get_relevant_documents(query)
print("\nMost relevant chunk:\n", results[0].page_content[:300])

