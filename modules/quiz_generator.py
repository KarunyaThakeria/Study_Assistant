
import json

from modules.config import get_client, MODEL
from modules.vector_store import search_similar_chunks


def retrieve_quiz_context(
    index,
    chunks,
    topic,
    weak_topics=None,
    top_k=10
):
    weak_topics = weak_topics or []

    retrieved = []

    # Retrieve topic-specific chunks
    if topic and topic.strip():
        topic_results = search_similar_chunks(
            query=(
                f"Definitions, concepts, explanations and examples "
                f"about {topic}"
            ),
            index=index,
            chunks=chunks,
            top_k=top_k
        )

        retrieved.extend(topic_results)

    # Retrieve separate chunks for each weak topic
    for weak_topic in weak_topics:
        weak_results = search_similar_chunks(
            query=(
                f"Important concepts, explanations and examples "
                f"about {weak_topic}"
            ),
            index=index,
            chunks=chunks,
            top_k=3
        )

        retrieved.extend(weak_results)

    # Remove duplicate chunks
    unique_chunks = []
    seen = set()

    for chunk in retrieved:
        key = (
            chunk["page_number"],
            chunk["chunk_number"]
        )

        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return unique_chunks[:15]


def generate_quiz(
    index,
    chunks,
    topic,
    question_type="MCQ",
    difficulty="Medium",
    num_questions=5,
    weak_topics=None
):
    weak_topics = weak_topics or []

    retrieved = retrieve_quiz_context(
        index=index,
        chunks=chunks,
        topic=topic,
        weak_topics=weak_topics
    )

    if not retrieved:
        return []

    context = "\n\n".join([
        f"[Page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in retrieved
    ])

    if question_type == "MCQ":
        format_instructions = """
        Each question must have exactly four options.
        Provide one correct answer.
        The correct_answer must exactly match one option.
        Leave reference_answer as an empty string.
        """

    else:
        format_instructions = """
        Do not provide options.
        Provide a detailed reference_answer.
        Set options to an empty list.
        Set correct_answer to an empty string.
        """

    weak_topic_instruction = ""

    if weak_topics:
        weak_topic_instruction = (
            "Give additional attention to these weak topics: "
            + ", ".join(weak_topics)
        )

    prompt = f"""
    You are an academic quiz generator.

    Generate exactly {num_questions} questions.

    Topic: {topic}
    Question Type: {question_type}
    Difficulty: {difficulty}

    {weak_topic_instruction}

    Use ONLY the provided study material.

    Requirements:
    1. Questions must test understanding, not just memorization.
    2. Match the requested difficulty.
    3. Avoid duplicate questions.
    4. Do not invent facts outside the material.
    5. Include a short explanation for each answer.
    6. Assign a topic to every question.
    7. Include source_pages using only page numbers
       visible in the supplied context.

    {format_instructions}

    Return valid JSON in exactly this structure:

    {{
      "questions": [
        {{
          "question": "Question text",
          "type": "{question_type}",
          "difficulty": "{difficulty}",
          "topic": "Specific topic",
          "options": [],
          "correct_answer": "",
          "reference_answer": "Answer for descriptive questions",
          "explanation": "Explanation",
          "source_pages": []
        }}
      ]
    }}

    Study Material:
    {context}
    """

    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate accurate educational quizzes "
                    "strictly grounded in supplied study material. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.4
    )

    result = json.loads(
        response.choices[0].message.content
    )

    questions = result.get("questions", [])

    # Keep source pages within the retrieved material
    allowed_pages = sorted({
        chunk["page_number"] for chunk in retrieved
    })

    cleaned_questions = []

    for question in questions[:num_questions]:
        question["options"] = question.get("options") or []

        question["source_pages"] = [
            page for page in question.get("source_pages", [])
            if page in allowed_pages
        ]

        if not question["source_pages"]:
            question["source_pages"] = allowed_pages

        question["topic"] = (
            question.get("topic") or topic or "General"
        )

        question["question_id"] = len(cleaned_questions) + 1

        cleaned_questions.append(question)

    return cleaned_questions


def evaluate_mcq(student_answer, question):
    correct_answer = question.get("correct_answer", "")

    is_correct = (
        student_answer.strip().lower()
        == correct_answer.strip().lower()
    )

    return {
        "is_correct": is_correct,
        "score": 10 if is_correct else 0,
        "feedback": (
            "Correct answer!"
            if is_correct
            else f"Incorrect. Correct answer: {correct_answer}"
        )
    }


def evaluate_descriptive_answer(
    student_answer,
    question
):
    client = get_client()

    prompt = f"""
    Evaluate the student's answer based on the reference answer.

    Question:
    {question['question']}

    Reference Answer:
    {question.get('reference_answer', '')}

    Student Answer:
    {student_answer}

    Evaluate conceptual correctness, important points,
    and understanding.

    Give a score from 0 to 10.

    Return JSON:
    {{
        "score": 0,
        "max_score": 10,
        "is_correct": false,
        "feedback": "Feedback for the student",
        "missing_concepts": [],
        "improved_answer": "A better answer"
    }}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fair academic evaluator. "
                    "Evaluate answers based on the supplied "
                    "reference answer, not unrelated knowledge. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    result = json.loads(
        response.choices[0].message.content
    )

    score = max(
        0,
        min(10, float(result.get("score", 0)))
    )

    return {
        "score": score,
        "max_score": 10,
        "is_correct": score >= 6,
        "feedback": result.get("feedback", ""),
        "missing_concepts": result.get(
            "missing_concepts", []
        ),
        "improved_answer": result.get(
            "improved_answer", ""
        )
    }
