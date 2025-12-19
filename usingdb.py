import chromadb
from chromadb.config import Settings
#to have an accessible database it will not lose memory if program stops uses duckdb
chroma_client = chromadb.Client(
    settings=Settings(
        persist_directory="./chroma_db"
    )
)
collection = chroma_client.create_collection(name="my_collection")
from sentence_transformers import SentenceTransformer

#installed manually inbuilt.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#using api calls
# import chromadb.utils.embedding_functions as embedding_functions
# huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
#     api_key="hf_----",
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


articles=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
# chunks = chunk_text(articles) # use this only when the articles isnt a list but a string.
ids = [f"doc1_chunk_{i}" for i in range(len(articles))]
# vectors=huggingface_ef(articles)
vectors = model.encode(articles).tolist()

collection.add(
    ids=ids,
    documents=articles,
    embeddings=vectors,
)

query_texts=["This is a query document about hawaii"]
query_embeddings = model.encode(query_texts).tolist()

results = collection.query(
    query_embeddings=query_embeddings, # Chroma will embed this for you
    n_results=1 # how many results to return
)
print(results)


#This is to see the data stored in the database
data = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

print(data)

