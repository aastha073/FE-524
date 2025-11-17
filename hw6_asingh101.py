#!/usr/bin/env python3
"""
Homework 6: Semantic Chunking and Embeddings

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
CHUNK_SIZE = 800  # characters
OVERLAP = 200     # characters
EMBEDDING_MODEL = "text-embedding-3-small"
OUTPUT_FILE = "knowledge_base.json"


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
        if len(chunk.strip()) > 0:  # Skip empty chunks
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
    """Compute embeddings for each chunk using OpenAI API."""
    embeddings = []
    
    print(f"Computing embeddings for {len(chunks)} chunks...")
    
    # Process in batches to avoid rate limits
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


def save_knowledge_base(chunks, embeddings, filename=OUTPUT_FILE):
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
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"\nKnowledge base saved to {filename}")


def main():
    print("=" * 60)
    print("Homework 6: Semantic Chunking and Embeddings")
    print("=" * 60)
    
    # Step 1: Download documents
    print("\n[1] Downloading documents...")
    documents = download_documents(DOCUMENT_URLS)
    total_text = "\n\n".join(documents)
    print(f"Downloaded {len(documents)} documents ({len(total_text):,} characters)")
    
    # Step 2: Semantic chunking
    print(f"\n[2] Semantic chunking (size={CHUNK_SIZE}, overlap={OVERLAP})...")
    chunks = semantic_chunk(total_text)
    print(f"Created {len(chunks)} semantic chunks")
    
    # Step 3: Compute embeddings
    print(f"\n[3] Computing embeddings...")
    embeddings = compute_embeddings(chunks)
    
    # Verify dimensions
    embedding_dim = len(embeddings[0])
    print(f"Embedding dimension: {embedding_dim}")
    
    # Step 4: Save to file
    print(f"\n[4] Saving knowledge base...")
    save_knowledge_base(chunks, embeddings)
    
    # Summary statistics
    avg_length = sum(len(c) for c in chunks) / len(chunks)
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Average chunk length: {avg_length:.1f} chars")
    print(f"  Output file: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
