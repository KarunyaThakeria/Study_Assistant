
import faiss
import numpy as np

from modules.config import get_client, EMBEDDING_MODEL


# --------------------------------------------------
# GENERATE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(texts):

    if not texts:
        return np.empty((0, 0), dtype="float32")

    client = get_client()

    embeddings = []

    batch_size = 100

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )

        embeddings.extend([
            item.embedding for item in response.data
        ])

    return np.array(embeddings, dtype="float32")


# --------------------------------------------------
# CREATE FAISS VECTOR STORE
# --------------------------------------------------

def create_vector_store(chunks):

    if not chunks:
        raise ValueError("No chunks available for indexing.")

    texts = [
        chunk["text"] for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Return BOTH index and chunks
    return index, chunks


# --------------------------------------------------
# SEARCH SIMILAR CHUNKS
# --------------------------------------------------

def search_similar_chunks(
    query,
    index,
    chunks,
    top_k=4
):

    if index is None or not chunks:
        return []

    if index.ntotal == 0:
        return []

    query_embedding = generate_embeddings([query])

    faiss.normalize_L2(query_embedding)

    k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        result = chunks[idx].copy()

        result["similarity_score"] = float(score)

        results.append(result)

    return results


# --------------------------------------------------
# SEARCH WITH DOCUMENT FILTER
# --------------------------------------------------

def search_document_chunks(
    query,
    index,
    chunks,
    selected_document=None,
    top_k=4
):

    if index is None or not chunks:
        return []

    # Search all documents
    if selected_document is None:

        return search_similar_chunks(
            query=query,
            index=index,
            chunks=chunks,
            top_k=top_k
        )

    # Filter chunks belonging to selected PDF
    filtered_chunks = [
        chunk for chunk in chunks
        if chunk.get("source") == selected_document
    ]

    if not filtered_chunks:
        return []

    # Build a temporary index for the selected PDF.
    # This ensures that the results belong only to
    # the selected document.

    texts = [
        chunk["text"] for chunk in filtered_chunks
    ]

    embeddings = generate_embeddings(texts)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    filtered_index = faiss.IndexFlatIP(dimension)

    filtered_index.add(embeddings)

    return search_similar_chunks(
        query=query,
        index=filtered_index,
        chunks=filtered_chunks,
        top_k=top_k
    )
