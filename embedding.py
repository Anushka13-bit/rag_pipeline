from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------
# Setup
# ------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------
# Load your document
# ------------------------------
sentences = ["I love a car", "I have a very big family"]

# IMPORTANT: splitter expects a STRING, not a list
document = " ".join(sentences)

# ------------------------------
# Split document into chunks
# ------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,    # max characters per chunk
    chunk_overlap=50   # overlap between chunks
)

chunks = splitter.split_text(document)
print(f"Total chunks created: {len(chunks)}")

# ------------------------------
# Encode chunks with SentenceTransformer
# ------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True
).astype("float32")

print("Embeddings shape:", embeddings.shape)

# ------------------------------
# Build FAISS index
# ------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Number of vectors in index:", index.ntotal)

# ------------------------------
# Groq client setup
# ------------------------------
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ------------------------------
# User query loop (ask until you stop)
# ------------------------------
while True:
    query = input("\nEnter your question (type 'stop' to exit): ")

    if query.lower() == "stop":
        print("Exiting RAG chat.")
        break

    # ------------------------------
    # Encode query
    # ------------------------------
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # ------------------------------
    # Retrieve relevant chunks
    # ------------------------------
    k = min(5, len(chunks))
    distances, indices = index.search(query_embedding, k)

    context = "\n".join([chunks[i] for i in indices[0]])

    print("\nRetrieved context:\n", context)

    # ------------------------------
    # Prepare prompt
    # ------------------------------
    prompt = f"""
You are an expert assistant.

You are given reference information.
Your task is to:
- Understand the context
- Combine related ideas
- Explain in your own words
- Do NOT copy sentences verbatim

Context:
{context}

Question:
{query}

Give a clear, well-reasoned explanation.
"""

    # ------------------------------
    # Call Groq LLM
    # ------------------------------
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    print("\n🧠 Answer:\n")
    print(response.choices[0].message.content)
