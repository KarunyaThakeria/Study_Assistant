
from modules.database import get_topic_performance


def get_weak_topics():
    performance = get_topic_performance()

    weak_topics = []

    for item in performance:
        if (
            item["attempts"] >= 2
            and item["accuracy"] < 60
        ):
            weak_topics.append(item["topic"])

    return weak_topics


def recommend_difficulty(topic):
    performance = get_topic_performance()

    for item in performance:
        if item["topic"].lower() == topic.lower():

            if item["accuracy"] < 60:
                return "Easy"

            elif item["accuracy"] < 80:
                return "Medium"

            else:
                return "Hard"

    return "Medium"


def get_adaptive_quiz_settings():
    weak_topics = get_weak_topics()

    if weak_topics:
        return {
            "weak_topics": weak_topics,
            "difficulty": "Easy"
        }

    return {
        "weak_topics": [],
        "difficulty": "Medium"
    }
