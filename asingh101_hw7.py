#!/usr/bin/env python3
"""
Homework 7: Retrieval-Augmented Generation (RAG)

"""

import os
import json
import urllib.request
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Document URLs to download
DOCUMENT_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
]

# Configuration
CHUNK_SIZE = 800
OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
TOP_K = 5  # Number of most relevant chunks to retrieve


def download_documents(urls):
    """Download documents from URLs."""
    documents = []
    for url in urls:
        print(f"Downloading: {url}")
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
            documents.append(text)
    return documents


def semantic_chunk(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks using range() with stride pattern from Week 7, Slide 11."""
    chunks = []
    stride = chunk_size - overlap
    
    # Using range(start, end, step) pattern from Week 7, Slides 9-11
    for i in range(0, len(text), stride):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    
    return chunks


def embed(text):
    """Compute embedding for a single text string (Week 6, Slide 10 pattern)."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float"
    )
    return resp.data[0].embedding


def compute_embeddings(chunks):
    """Compute embeddings for each chunk."""
    embeddings = []
    
    print(f"Computing embeddings for {len(chunks)} chunks...")
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float"
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Processed {i + len(batch)} chunks...")
    
    return embeddings


def save_knowledge_base(chunks, embeddings):
    """Save chunks and embeddings to JSON file."""
    data = {
        "chunks": chunks,
        "embeddings": embeddings,
        "metadata": {
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "num_chunks": len(chunks)
        }
    }
    
    with open(KNOWLEDGE_BASE_FILE, 'w') as f:
        json.dump(data, f)
    
    print(f"Knowledge base saved to {KNOWLEDGE_BASE_FILE}")


def load_knowledge_base():
    """Load knowledge base from file."""
    with open(KNOWLEDGE_BASE_FILE, 'r') as f:
        data = json.load(f)
    return data["chunks"], data["embeddings"]


def initialize_knowledge_base():
    """Initialize knowledge base if it doesn't exist."""
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"Knowledge base found: {KNOWLEDGE_BASE_FILE}")
        print("Loading existing knowledge base...")
        return load_knowledge_base()
    else:
        print("Knowledge base not found. Creating new one...")
        print("\n[1] Downloading documents...")
        documents = download_documents(DOCUMENT_URLS)
        total_text = "\n\n".join(documents)
        print(f"Downloaded {len(documents)} documents ({len(total_text):,} characters)")
        
        print(f"\n[2] Semantic chunking (size={CHUNK_SIZE}, overlap={OVERLAP})...")
        chunks = semantic_chunk(total_text)
        print(f"Created {len(chunks)} semantic chunks")
        
        print(f"\n[3] Computing embeddings...")
        embeddings = compute_embeddings(chunks)
        
        print(f"\n[4] Saving knowledge base...")
        save_knowledge_base(chunks, embeddings)
        
        return chunks, embeddings


def compute_query_embedding(query):
    """Compute embedding for the query using embed() helper."""
    return embed(query)


def dot_product(v1, v2):
    """Compute dot product of two vectors (Week 7, Slide 13)."""
    return sum(a * b for a, b in zip(v1, v2))


def retrieve_relevant_chunks(query_embedding, chunk_embeddings, chunks, k=TOP_K):
    """Retrieve top-k most similar chunks using cosine similarity."""
    # Compute cosine similarity (dot product for normalized embeddings - Week 7, Slide 21)
    similarities = []
    for chunk_emb in chunk_embeddings:
        similarity = dot_product(query_embedding, chunk_emb)
        similarities.append(similarity)
    
    # Get top-k indices (Week 7, Slide 22)
    # Create list of (index, similarity) pairs
    indexed_similarities = list(enumerate(similarities))
    # Sort by similarity (descending)
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)
    # Get top k indices
    topk_indices = [idx for idx, _ in indexed_similarities[:k]]
    
    # Get top-k chunks and their similarity scores
    retrieved_chunks = [(chunks[i], similarities[i]) for i in topk_indices]
    
    return retrieved_chunks, topk_indices


def create_augmented_prompt(query, retrieved_chunks, topk_indices, chunks):
    """Create developer prompt with retrieved knowledge (Week 7, Slide 23)."""
    context = []
    
    for i, idx in enumerate(topk_indices):
        context.append(f"""
<retrieved_knowledge id="document-{i}">
{chunks[idx]}
</retrieved_knowledge>
""")
    
    developer_prompt = f"""Use the following to answer questions.
{"\n".join(context)}
"""
    
    return developer_prompt


def send_rag_query(query, chunks, embeddings):
    """Send a RAG query to the Responses API."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}")
    
    # Step 1: Compute query embedding
    print("\n[1] Computing query embedding...")
    query_embedding = compute_query_embedding(query)
    
    # Step 2: Retrieve relevant chunks
    print(f"[2] Retrieving top {TOP_K} relevant chunks...")
    retrieved_chunks, topk_indices = retrieve_relevant_chunks(
        query_embedding, embeddings, chunks, k=TOP_K
    )
    
    # Display similarity scores
    print("\nRetrieved chunks (with similarity scores):")
    for i, (chunk, score) in enumerate(retrieved_chunks):
        preview = chunk[:100].replace('\n', ' ')
        print(f"  {i+1}. Score: {score:.4f} | Preview: {preview}...")
    
    # Step 3: Create augmented prompt
    print("\n[3] Creating augmented prompt...")
    developer_prompt = create_augmented_prompt(query, retrieved_chunks, topk_indices, chunks)
    
    # Step 4: Send to Responses API
    print("[4] Sending RAG query to Responses API...")
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": query}
        ]
    )
    
    # Extract answer
    answer = response.output[0].content
    
    # Display results
    print(f"\n{'=' * 60}")
    print("ANSWER:")
    print(f"{'=' * 60}")
    print(answer)
    print(f"{'=' * 60}\n")
    
    return answer


def main():
    print("=" * 60)
    print("Homework 7: Retrieval-Augmented Generation")
    print("=" * 60)
    
    # Initialize knowledge base (loads if exists, creates if not)
    chunks, embeddings = initialize_knowledge_base()
    print(f"\nKnowledge base ready: {len(chunks)} chunks")
    
    # Example RAG queries
    queries = [
        "What is the relationship between Elizabeth and Mr. Darcy?",
        "Describe Alice's adventures in Wonderland.",
        "What are the main themes in these texts?"
    ]
    
    # Run first query
    query = queries[0]
    send_rag_query(query, chunks, embeddings)
    
    print("\nAdditional example queries you could try:")
    for i, q in enumerate(queries[1:], 2):
        print(f"  {i}. {q}")


if __name__ == "__main__":
    main()
