
from modules.config import get_client, MODEL

from modules.vector_store import (
    search_similar_chunks,
    search_document_chunks
)


# --------------------------------------------------
# ANSWER USER QUESTION
# --------------------------------------------------

def answer_question(
    question,
    index,
    chunks,
    top_k=4,
    selected_document=None
):

    # Retrieve relevant chunks
    if selected_document is None:

        retrieved_chunks = search_similar_chunks(
            query=question,
            index=index,
            chunks=chunks,
            top_k=top_k
        )

    else:

        retrieved_chunks = search_document_chunks(
            query=question,
            index=index,
            chunks=chunks,
            selected_document=selected_document,
            top_k=top_k
        )

    if not retrieved_chunks:

        return {
            "answer": "No relevant information was found in your notes.",
            "sources": []
        }

    # Prepare context
    context_parts = []

    for chunk in retrieved_chunks:

        filename = chunk.get(
            "source",
            "Unknown PDF"
        )

        page_number = chunk.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
            Source: {filename}
            Page: {page_number}

            Content:
            {chunk["text"]}
            """
        )

    context = "\n\n".join(context_parts)

    # System prompt
    system_prompt = """
    You are a helpful Personal Study Assistant.

    Answer questions using only the provided study material.

    Instructions:

    1. Explain concepts in simple student-friendly language.
    2. Use examples wherever helpful.
    3. Do not invent information.
    4. If the answer is unavailable in the context,
       clearly say that it is not covered in the uploaded notes.
    5. Mention relevant PDF filenames and page numbers.
    6. Organize longer answers using headings and bullet points.
    """

    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
                Study Material:

                {context}

                Student Question:

                {question}

                Answer based only on the provided study material.
                """
            }
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    # Prepare source references
    sources = []

    for chunk in retrieved_chunks:

        sources.append({

            "source": chunk.get(
                "source",
                "Unknown PDF"
            ),

            "page_number": chunk.get(
                "page_number",
                "Unknown"
            ),

            "similarity_score": chunk.get(
                "similarity_score",
                0
            ),

            "text": chunk["text"]

        })

    return {
        "answer": answer,
        "sources": sources
    }
