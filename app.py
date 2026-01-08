import streamlit as st
from classifier import predict_difficulty

st.set_page_config(
    page_title="AutoJudge",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AutoJudge")
st.write("Predict programming problem difficulty")

title = st.text_input(
    "Problem Title",
    placeholder="e.g. Shortest Path in Graph"
)

problem_desc = st.text_area(
    "Problem Description",
    height=200
)

input_desc = st.text_area(
    "Input Description",
    height=120
)

output_desc = st.text_area(
    "Output Description",
    height=120
)

if st.button("Predict Difficulty"):
    if problem_desc.strip() == "":
        st.warning("Please enter the problem description.")
    else:
        difficulty, score = predict_difficulty(
            title,
            problem_desc,
            input_desc,
            output_desc
        )

        st.success("Prediction Complete")
        st.markdown(f"### 📌 Difficulty Class: **{difficulty.upper()}**")
        st.markdown(f"### 🔢 Difficulty Score: **{score} / 100**")
        st.progress(score / 100)
