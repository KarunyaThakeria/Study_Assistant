
import os

import streamlit as st
import pandas as pd

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

from modules.pdf_processor import process_pdf

from modules.vector_store import create_vector_store

from modules.rag_pipeline import answer_question

from modules.summarizer import generate_summary

from modules.quiz_generator import (
    generate_quiz,
    evaluate_mcq,
    evaluate_descriptive_answer
)

from modules.database import (
    initialize_database,
    save_attempt,
    get_all_attempts,
    get_topic_performance
)

from modules.adaptive_learning import (
    get_weak_topics,
    recommend_difficulty
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Personal Study Assistant",
    page_icon="📚",
    layout="wide"
)

os.makedirs("data/uploaded_pdfs", exist_ok=True)

initialize_database()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

default_values = {
    "index": None,
    "chunks": [],
    "pages": [],
    "document_names": [],
    "quiz": [],
    "quiz_results": None
}

for key, value in default_values.items():

    if key not in st.session_state:

        st.session_state[key] = value


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def document_uploaded():

    return (
        st.session_state.index is not None
        and len(st.session_state.chunks) > 0
    )


def clear_quiz():

    st.session_state.quiz = []

    st.session_state.quiz_results = None


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📚 Study Assistant")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Ask AI",
        "Smart Notes",
        "Adaptive Quiz",
        "Learning Progress"
    ]
)

st.sidebar.divider()


# --------------------------------------------------
# MULTIPLE PDF UPLOAD
# --------------------------------------------------

st.sidebar.subheader("Upload Study Materials")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    if st.sidebar.button(
        "Process All PDFs",
        use_container_width=True
    ):

        all_chunks = []

        all_pages = []

        processed_files = []

        failed_files = []

        progress = st.sidebar.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            try:

                st.sidebar.write(
                    f"Processing: {uploaded_file.name}"
                )

                # Process individual PDF
                pages, chunks = process_pdf(uploaded_file)

                # Attach filename to page metadata
                for page_data in pages:

                    page_data["source"] = uploaded_file.name

                # Attach filename to chunk metadata
                for chunk in chunks:

                    chunk["source"] = uploaded_file.name

                all_pages.extend(pages)

                all_chunks.extend(chunks)

                processed_files.append(uploaded_file.name)

                # Save PDF locally
                file_path = os.path.join(
                    "data",
                    "uploaded_pdfs",
                    os.path.basename(uploaded_file.name)
                )

                with open(file_path, "wb") as f:

                    f.write(uploaded_file.getvalue())

            except Exception as e:

                failed_files.append(uploaded_file.name)

                st.sidebar.error(
                    f"Error processing {uploaded_file.name}: {e}"
                )

            progress.progress(
                (i + 1) / len(uploaded_files)
            )

        # Create combined vector database
        if all_chunks:

            try:

                with st.spinner(
                    "Creating combined FAISS vector database..."
                ):

                    # IMPORTANT:
                    # create_vector_store returns two values
                    index, processed_chunks = create_vector_store(
                        all_chunks
                    )

                # Store actual FAISS index, not tuple
                st.session_state.index = index

                st.session_state.chunks = processed_chunks

                st.session_state.pages = all_pages

                st.session_state.document_names = processed_files

                clear_quiz()

                st.sidebar.success(
                    f"Successfully processed {len(processed_files)} PDFs!"
                )

                st.sidebar.caption(
                    f"{len(all_pages)} pages | "
                    f"{len(processed_chunks)} chunks"
                )

            except Exception as e:

                st.sidebar.error(
                    f"Could not create vector database: {e}"
                )

        else:

            st.sidebar.error(
                "No PDFs could be processed successfully."
            )

        if failed_files:

            st.sidebar.warning(
                "Failed files: " + ", ".join(failed_files)
            )


# --------------------------------------------------
# DISPLAY LOADED DOCUMENTS
# --------------------------------------------------

if document_uploaded():

    st.sidebar.success(
        f"{len(st.session_state.document_names)} PDFs loaded"
    )

    for filename in st.session_state.document_names:

        st.sidebar.caption(
            f"📄 {filename}"
        )

    st.sidebar.caption(
        f"{len(st.session_state.pages)} pages | "
        f"{len(st.session_state.chunks)} chunks"
    )

else:

    st.sidebar.info(
        "Upload and process PDFs to begin."
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("📚 Personal Study Assistant")

    st.write(
        "Your AI-powered learning companion. "
        "Upload lecture notes, ask questions, "
        "generate summaries and practice quizzes."
    )

    attempts = get_all_attempts()

    total_attempts = len(attempts)

    if total_attempts:

        correct = sum(
            item["is_correct"]
            for item in attempts
        )

        accuracy = round(
            (correct / total_attempts) * 100,
            2
        )

    else:

        accuracy = 0

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents Loaded",
        len(st.session_state.document_names)
    )

    col2.metric(
        "Quiz Attempts",
        total_attempts
    )

    col3.metric(
        "Overall Accuracy",
        f"{accuracy}%"
    )

    st.divider()

    st.subheader("What can you do?")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 💬 Ask AI")

        st.write(
            "Ask questions about your uploaded notes "
            "and get context-based answers."
        )

        st.markdown("### 📝 Smart Notes")

        st.write(
            "Generate short summaries, detailed explanations "
            "and exam-oriented notes."
        )

    with col2:

        st.markdown("### 🧠 Adaptive Quiz")

        st.write(
            "Generate MCQs, conceptual and descriptive "
            "questions based on your study material."
        )

        st.markdown("### 📊 Learning Progress")

        st.write(
            "Track quiz performance and identify "
            "topics that need more practice."
        )


# --------------------------------------------------
# ASK AI
# --------------------------------------------------

elif page == "Ask AI":

    st.title("💬 Ask AI")

    st.write(
        "Ask questions about all your uploaded study materials."
    )

    if not document_uploaded():

        st.warning(
            "Please upload and process PDFs first."
        )

    else:

        selected_document = st.selectbox(
            "Search in",
            ["All Documents"] + st.session_state.document_names
        )

        question = st.text_area(
            "Enter your question",
            placeholder="Explain self-attention in simple words..."
        )

        top_k = st.slider(
            "Number of relevant chunks",
            min_value=2,
            max_value=10,
            value=4
        )

        if st.button("Get Answer"):

            if not question.strip():

                st.warning("Please enter a question.")

            else:

                try:

                    with st.spinner("Searching your notes..."):

                        # Convert All Documents to None
                        document_filter = (
                            None
                            if selected_document == "All Documents"
                            else selected_document
                        )

                        result = answer_question(
                            question=question,
                            index=st.session_state.index,
                            chunks=st.session_state.chunks,
                            top_k=top_k,
                            selected_document=document_filter
                        )

                    st.subheader("Answer")

                    st.markdown(result["answer"])

                    st.divider()

                    st.subheader("Sources")

                    if result["sources"]:

                        for source in result["sources"]:

                            filename = source.get(
                                "source",
                                "Unknown PDF"
                            )

                            with st.expander(
                                f"{filename} | "
                                f"Page {source['page_number']}"
                            ):

                                st.write(source["text"])

                                st.caption(
                                    "Similarity: "
                                    f"{source['similarity_score']:.3f}"
                                )

                    else:

                        st.info("No sources found.")

                except Exception as e:

                    st.error(f"Error: {str(e)}")


# --------------------------------------------------
# SMART NOTES
# --------------------------------------------------

elif page == "Smart Notes":

    st.title("📝 Smart Notes")

    st.write(
        "Generate topic-specific notes using your uploaded PDFs."
    )

    if not document_uploaded():

        st.warning(
            "Please upload and process PDFs first."
        )

    else:

        topic = st.text_input(
            "Enter topic to summarize",
            placeholder="e.g. Convolutional Neural Networks"
        )

        summary_type = st.selectbox(
            "Select summary format",
            [
                "Short Summary",
                "Detailed Summary",
                "Exam Notes"
            ]
        )

        if st.button("Generate Notes"):

            if not topic.strip():

                st.warning("Please enter a topic.")

            else:

                try:

                    with st.spinner(
                        "Generating your notes..."
                    ):

                        summary = generate_summary(
                            topic=topic,
                            index=st.session_state.index,
                            chunks=st.session_state.chunks,
                            summary_type=summary_type
                        )

                    st.subheader("Generated Notes")

                    st.markdown(summary)

                except Exception as e:

                    st.error(f"Error: {str(e)}")


# --------------------------------------------------
# ADAPTIVE QUIZ
# --------------------------------------------------

elif page == "Adaptive Quiz":

    st.title("🧠 Adaptive Quiz Generator")

    st.write(
        "Practice questions generated from your study materials."
    )

    if not document_uploaded():

        st.warning(
            "Please upload and process PDFs first."
        )

    else:

        quiz_mode = st.radio(
            "Quiz Mode",
            [
                "Custom Quiz",
                "Adaptive Quiz"
            ],
            horizontal=True
        )

        weak_topics = get_weak_topics()

        topic = ""

        selected_weak_topics = []

        focus = None

        if quiz_mode == "Custom Quiz":

            topic = st.text_input(
                "Enter quiz topic",
                placeholder="e.g. Neural Networks"
            )

        else:

            if not weak_topics:

                st.info(
                    "No weak topics detected yet. "
                    "Attempt at least two questions on a topic "
                    "to identify areas for improvement."
                )

            else:

                focus = st.selectbox(
                    "Choose a weak topic",
                    ["All weak topics"] + weak_topics
                )

                if focus == "All weak topics":

                    selected_weak_topics = weak_topics

                    topic = "All weak topics"

                else:

                    selected_weak_topics = [focus]

                    topic = focus

        question_type = st.selectbox(
            "Question Type",
            [
                "MCQ",
                "Conceptual",
                "Descriptive"
            ]
        )

        default_difficulty = "Medium"

        if quiz_mode == "Adaptive Quiz" and weak_topics:

            if focus == "All weak topics":

                default_difficulty = "Easy"

            else:

                default_difficulty = recommend_difficulty(focus)

        difficulty_options = [
            "Easy",
            "Medium",
            "Hard"
        ]

        difficulty = st.selectbox(
            "Difficulty",
            difficulty_options,
            index=difficulty_options.index(default_difficulty)
        )

        num_questions = st.slider(
            "Number of Questions",
            min_value=3,
            max_value=10,
            value=5
        )

        can_generate = (

            quiz_mode == "Custom Quiz"
            and bool(topic.strip())

        ) or (

            quiz_mode == "Adaptive Quiz"
            and bool(weak_topics)

        )

        if st.button(
            "Generate Quiz",
            disabled=not can_generate
        ):

            try:

                with st.spinner(
                    "Generating your quiz..."
                ):

                    questions = generate_quiz(
                        index=st.session_state.index,
                        chunks=st.session_state.chunks,
                        topic=topic,
                        question_type=question_type,
                        difficulty=difficulty,
                        num_questions=num_questions,
                        weak_topics=selected_weak_topics
                    )

                if not questions:

                    st.warning(
                        "No questions generated. Try another topic."
                    )

                else:

                    st.session_state.quiz = questions

                    st.session_state.quiz_results = None

                    st.success(
                        f"{len(questions)} questions generated!"
                    )

            except Exception as e:

                st.error(f"Error: {str(e)}")


    # --------------------------------------------------
    # DISPLAY QUIZ
    # --------------------------------------------------

    if st.session_state.quiz:

        st.divider()

        st.subheader("Your Quiz")

        with st.form("quiz_submission_form"):

            student_answers = []

            for i, question in enumerate(
                st.session_state.quiz
            ):

                st.markdown(
                    f"**Q{i + 1}. {question['question']}**"
                )

                st.caption(
                    f"Topic: {question.get('topic', 'General')} "
                    f"| Difficulty: "
                    f"{question.get('difficulty', 'Medium')}"
                )

                if question.get("source_pages"):

                    st.caption(
                        "Source pages: "
                        + ", ".join(
                            str(p)
                            for p in question["source_pages"]
                        )
                    )

                if question.get("type") == "MCQ":

                    options = question.get("options", [])

                    answer = st.selectbox(
                        "Choose your answer",
                        ["-- Select an answer --"] + options,
                        key=f"quiz_answer_{i}"
                    )

                else:

                    answer = st.text_area(
                        "Write your answer",
                        key=f"quiz_answer_{i}"
                    )

                student_answers.append(answer)

                st.divider()

            submitted = st.form_submit_button(
                "Submit Quiz",
                use_container_width=True
            )

        if submitted:

            unanswered = []

            for i, answer in enumerate(student_answers):

                if (
                    not answer
                    or not answer.strip()
                    or answer == "-- Select an answer --"
                ):

                    unanswered.append(i + 1)

            if unanswered:

                st.warning(
                    "Please answer all questions. "
                    "Unanswered: "
                    + ", ".join(map(str, unanswered))
                )

            else:

                results = []

                try:

                    with st.spinner("Evaluating your answers..."):

                        for question, student_answer in zip(
                            st.session_state.quiz,
                            student_answers
                        ):

                            if question.get("type") == "MCQ":

                                evaluation = evaluate_mcq(
                                    student_answer,
                                    question
                                )

                            else:

                                evaluation = evaluate_descriptive_answer(
                                    student_answer,
                                    question
                                )

                            save_attempt(
                                question,
                                student_answer,
                                evaluation
                            )

                            results.append({

                                "question": question["question"],

                                "student_answer": student_answer,

                                "correct_answer": (
                                    question.get("correct_answer")
                                    or question.get("reference_answer", "")
                                ),

                                "evaluation": evaluation,

                                "explanation": question.get(
                                    "explanation",
                                    ""
                                ),

                                "topic": question.get(
                                    "topic",
                                    "General"
                                )
                            })

                    st.session_state.quiz_results = results

                except Exception as e:

                    st.error(
                        f"Could not complete quiz evaluation: {str(e)}"
                    )


    # --------------------------------------------------
    # DISPLAY QUIZ RESULTS
    # --------------------------------------------------

    if st.session_state.quiz_results is not None:

        results = st.session_state.quiz_results

        if results:

            total_score = sum(
                item["evaluation"]["score"]
                for item in results
            )

            max_score = len(results) * 10

            correct_count = sum(
                item["evaluation"]["is_correct"]
                for item in results
            )

            accuracy = round(
                (correct_count / len(results)) * 100,
                2
            )

            st.divider()

            st.subheader("📊 Quiz Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Score",
                f"{total_score:g}/{max_score}"
            )

            col2.metric(
                "Correct Answers",
                f"{correct_count}/{len(results)}"
            )

            col3.metric(
                "Accuracy",
                f"{accuracy}%"
            )

            for i, result in enumerate(results):

                evaluation = result["evaluation"]

                with st.expander(
                    f"Question {i + 1} — "
                    f"{'Correct' if evaluation['is_correct'] else 'Needs Practice'}"
                ):

                    st.write(result["question"])

                    st.write(
                        "**Your answer:**",
                        result["student_answer"]
                    )

                    if result["correct_answer"]:

                        st.write(
                            "**Reference answer:**",
                            result["correct_answer"]
                        )

                    st.write(
                        f"**Score:** {evaluation['score']}/10"
                    )

                    st.write(
                        "**Feedback:**",
                        evaluation["feedback"]
                    )

                    if result["explanation"]:

                        st.write(
                            "**Explanation:**",
                            result["explanation"]
                        )

                    if evaluation.get("missing_concepts"):

                        st.write(
                            "**Concepts to revise:**",
                            ", ".join(evaluation["missing_concepts"])
                        )

                    if evaluation.get("improved_answer"):

                        st.write(
                            "**Improved answer:**",
                            evaluation["improved_answer"]
                        )

            st.success(
                "Your quiz performance has been saved."
            )


# --------------------------------------------------
# LEARNING PROGRESS
# --------------------------------------------------

elif page == "Learning Progress":

    st.title("📊 Learning Progress")

    attempts = get_all_attempts()

    performance = get_topic_performance()

    weak_topics = get_weak_topics()

    if not attempts:

        st.info(
            "No quiz attempts yet. "
            "Complete a quiz to see your progress."
        )

    else:

        total_attempts = len(attempts)

        correct = sum(
            item["is_correct"]
            for item in attempts
        )

        accuracy = round(
            (correct / total_attempts) * 100,
            2
        )

        average_score = round(
            sum(item["score"] for item in attempts)
            / total_attempts,
            2
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Questions",
            total_attempts
        )

        col2.metric(
            "Overall Accuracy",
            f"{accuracy}%"
        )

        col3.metric(
            "Average Score",
            f"{average_score}/10"
        )

        st.divider()

        st.subheader("Topic-wise Performance")

        if performance:

            df = pd.DataFrame(performance)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            if "topic" in df.columns and "accuracy" in df.columns:

                st.bar_chart(
                    df.set_index("topic")["accuracy"]
                )

        st.divider()

        st.subheader("Topics to Improve")

        if weak_topics:

            for topic_name in weak_topics:

                st.warning(
                    f"Revise **{topic_name}** and attempt "
                    "more practice questions."
                )

        else:

            st.success(
                "No weak topics detected based on your current attempts."
            )

        st.divider()

        st.subheader("Recent Quiz History")

        history_df = pd.DataFrame(attempts)

        required_columns = [
            "timestamp",
            "topic",
            "question_type",
            "difficulty",
            "score",
            "is_correct"
        ]

        available_columns = [
            col for col in required_columns
            if col in history_df.columns
        ]

        st.dataframe(
            history_df[available_columns].head(20),
            use_container_width=True,
            hide_index=True
        )
