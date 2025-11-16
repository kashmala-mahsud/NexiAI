# filename: interview_app.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Setup ---
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=1.2,
    max_tokens=500
)

# --- Question Generator ---
question_prompt = ChatPromptTemplate.from_template("""
You are an expert interviewer. Generate {num_questions} unique, clear, field-specific questions.

Field: {field}
Tone: {tone}

Return format:
1. Question 1
2. Question 2
3. Question 3
""")
question_chain = question_prompt | llm | StrOutputParser()

# --- Evaluation Prompt ---
eval_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator. Evaluate the following interview answer for quality, relevance, and clarity.
Score it out of 10 and provide a brief comment.

Question: {question}
Answer: {answer}

Return format:
Score: <score out of 10>
Comment: <short comment>
""")
eval_chain = eval_prompt | llm | StrOutputParser()

# --- Streamlit App ---
st.title("🎤 AI Interview System")
st.sidebar.header("Interview Settings")

field = st.sidebar.text_input("Field", value="Data Science")
tone = st.sidebar.selectbox("Tone", options=["friendly", "professional", "simple"], index=0)
num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=20, value=5)

# --- Start / Reset Interview ---
if st.button("Start Interview") or "questions" not in st.session_state:
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.evaluations = []
    st.session_state.current_q = 0

    # Generate questions
    questions_text = question_chain.invoke({
        "field": field,
        "tone": tone,
        "num_questions": num_questions
    })

    questions = [
        q.strip() for q in questions_text.split("\n")
        if q.strip() and q.lstrip()[0].isdigit()
    ]
    st.session_state.questions = questions

# --- Exit Interview ---
if "questions" in st.session_state and st.session_state.questions:
    if st.button("Exit Interview"):
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.evaluations = []
        st.session_state.current_q = 0
        st.experimental_rerun()  # Safe to use here in latest Streamlit

# --- Interview Loop ---
if "questions" in st.session_state and st.session_state.questions:
    q_index = st.session_state.get("current_q", 0)

    if q_index < len(st.session_state.questions):
        st.subheader(f"Q{q_index+1}: {st.session_state.questions[q_index]}")
        answer = st.text_area("Your Answer:", key=f"answer_{q_index}")

        if st.button("Submit Answer", key=f"submit_{q_index}"):
            if answer.strip():
                # Evaluate answer
                eval_text = eval_chain.invoke({
                    "question": st.session_state.questions[q_index],
                    "answer": answer
                })

                # Parse evaluation
                try:
                    score_line = [line for line in eval_text.split("\n") if "Score:" in line][0]
                    score = float(score_line.split("Score:")[1].strip())
                except Exception:
                    score = 0

                comment_line = [line for line in eval_text.split("\n") if "Comment:" in line]
                comment = comment_line[0].split("Comment:")[1].strip() if comment_line else ""

                # Save answer + evaluation
                st.session_state.answers.append(answer)
                st.session_state.evaluations.append({"score": score, "comment": comment})
                st.session_state.current_q += 1
                st.experimental_rerun()

            else:
                st.warning("Please type an answer before submitting.")

    else:
        st.success("🎯 Interview Completed!")
        for i, (q, a, e) in enumerate(zip(
            st.session_state.questions,
            st.session_state.answers,
            st.session_state.evaluations
        )):
            st.markdown(f"**Q{i+1}: {q}**")
            st.markdown(f"- Your Answer: {a}")
            st.markdown(f"- Score: {e['score']} / 10")
            st.markdown(f"- Comment: {e['comment']}")
            st.markdown("---")

        if st.button("Save Interview to JSON"):
            filename = f"interview_{field.replace(' ','_')}.json"
            data = {
                "field": field,
                "tone": tone,
                "questions": st.session_state.questions,
                "answers": st.session_state.answers,
                "evaluations": st.session_state.evaluations
            }
            with open(filename, "w") as f:
                import json
                json.dump(data, f, indent=4)
            st.success(f"Interview saved to {filename}")
