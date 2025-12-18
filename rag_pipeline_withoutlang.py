from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredPowerPointLoader
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


path = input("Enter file or folder path: ").strip()

if not os.path.exists(path):
    raise FileNotFoundError(f"Path not found: {path}")

documents = []

if os.path.isdir(path):
    loaders = [
        DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(path, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader),
    ]
    for loader in loaders:
        documents.extend(loader.load())
else:
    if path.endswith((".txt", ".md")):
        documents = TextLoader(path).load()
    elif path.endswith(".pdf"):
        documents = PyPDFLoader(path).load()
    elif path.endswith(".pptx"):
        documents = UnstructuredPowerPointLoader(path).load()
    else:
        raise ValueError("Unsupported file type")

if not documents:
    raise ValueError("No documents loaded")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)
chunks = [doc.page_content for doc in docs]

print(f"Total chunks created: {len(chunks)}")


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index size:", index.ntotal)


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


print("\n🤖 Ask questions about the document. Type 'exit' to stop.\n")

while True:
    query = input("You: ").strip()

    if query.lower() in {"exit", "quit", "stop"}:
        print("👋 Exiting document Q&A")
        break

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    k = min(5, len(chunks))
    distances, indices = index.search(query_embedding, k)

    context = "\n".join(chunks[i] for i in indices[0])

    prompt = f"""
You are an expert assistant.

Use the provided context only.
Explain in your own words.
Do NOT copy sentences verbatim.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n🧠 Answer:\n")
    print(response.choices[0].message.content)
    print("\n" + "-" * 60)

