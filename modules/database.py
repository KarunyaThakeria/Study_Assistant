
import os
import sqlite3

DB_PATH = "data/study_assistant.db"


def get_connection():
    os.makedirs("data", exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            topic TEXT,
            question_type TEXT,
            difficulty TEXT,
            student_answer TEXT,
            correct_answer TEXT,
            is_correct INTEGER,
            score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_attempt(question, student_answer, evaluation):
    connection = get_connection()

    correct_answer = (
        question.get("correct_answer")
        or question.get("reference_answer", "")
    )

    connection.execute("""
        INSERT INTO quiz_attempts (
            question,
            topic,
            question_type,
            difficulty,
            student_answer,
            correct_answer,
            is_correct,
            score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        question["question"],
        question.get("topic", "General"),
        question.get("type", ""),
        question.get("difficulty", ""),
        student_answer,
        correct_answer,
        int(evaluation["is_correct"]),
        evaluation["score"]
    ))

    connection.commit()
    connection.close()


def get_all_attempts():
    connection = get_connection()

    rows = connection.execute("""
        SELECT * FROM quiz_attempts
        ORDER BY timestamp DESC
    """).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_topic_performance():
    connection = get_connection()

    rows = connection.execute("""
        SELECT
            topic,
            COUNT(*) AS attempts,
            SUM(is_correct) AS correct,
            AVG(score) AS average_score
        FROM quiz_attempts
        GROUP BY topic
        ORDER BY attempts DESC
    """).fetchall()

    connection.close()

    results = []

    for row in rows:
        attempts = row["attempts"]
        correct = row["correct"] or 0

        results.append({
            "topic": row["topic"],
            "attempts": attempts,
            "correct": correct,
            "accuracy": round(
                (correct / attempts) * 100, 2
            ),
            "average_score": round(
                row["average_score"] or 0, 2
            )
        })

    return results
