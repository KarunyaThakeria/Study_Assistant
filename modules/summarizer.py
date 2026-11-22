
from modules.config import get_client, MODEL
from modules.vector_store import search_similar_chunks


def generate_summary(
    topic,
    index,
    chunks,
    summary_type="Detailed Summary",
    top_k=8
):
    if not topic.strip():
        raise ValueError("Please enter a topic.")

    search_query = (
        f"Important concepts, definitions, explanations "
        f"and examples about {topic}"
    )

    retrieved = search_similar_chunks(
        query=search_query,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    if not retrieved:
        return "No relevant study material found for this topic."

    context = "\n\n".join([
        f"[Page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in retrieved
    ])

    prompts = {
        "Short Summary": """
        Create a concise summary covering the most important
        concepts and definitions.
        """,

        "Detailed Summary": """
        Create a detailed topic-wise explanation.
        Explain concepts clearly and include examples
        wherever the source material supports them.
        """,

        "Exam Notes": """
        Create exam-oriented notes with:
        - Important definitions
        - Key concepts
        - Important points
        - Examples
        - Differences between related concepts
        """
    }

    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are an academic tutor.

                Use only the retrieved study material.
                Do not invent facts.
                Organize the summary with headings and bullet points.
                Mention relevant source page numbers.
                If information is missing, do not make it up.
                """
            },
            {
                "role": "user",
                "content": f"""
                Topic: {topic}

                Study Material:
                {context}

                Task:
                {prompts[summary_type]}
                """
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
