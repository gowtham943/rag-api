import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

with open("profile.txt", "r") as f:
    knowledge_base = f.read()

chunks = [chunk.strip() for chunk in knowledge_base.split("\n\n") if chunk.strip()]

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Embedding function using Ollama
embedding_function = OllamaEmbeddingFunction(model_name="nomic-embed-text", url="http://localhost:11434")

# Create a collection for the knowledge base
collection = client.get_or_create_collection(name="knowledge_base", embedding_function=embedding_function)

# Add chunks to the collection - ChromaDB automatically generates embeddings
collection.add(
    ids=[f"chunk{i}" for i in range(len(chunks))],  # Unique ID for each chunk
    documents=chunks,  # The actual text content
    metadatas=[{"source": "profile", "chunk_index": i} for i in range(len(chunks))],
)

print(f"Added {len(chunks)} chunks to the 'personal_profile' collection.")
print("Knowledge base built successfully!")




