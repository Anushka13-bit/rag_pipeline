import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough


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

print(f"Loaded {len(documents)} documents")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)
print(f"Total chunks created: {len(docs)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


llm = Groq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant.

Use ONLY the context below.
Explain in your own words.
If the answer is not in the context, say "Not found in the document".

Context:
{context}

Question:
{question}

Answer:
"""
)



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

print("\nRAG Chatbot ready. Type 'exit' to stop.\n")

while True:
    query = input("You: ").strip()

    if query.lower() in {"exit", "quit", "stop"}:
        print("Exiting chatbot.")
        break

    response = chain.invoke(query)
    print("\n🤖 Answer:\n", response.content, "\n")
