import os

import pandas as pd
import streamlit as st
from openai import OpenAI

from agent import BASE_URL, DEFAULT_MODEL, DataAgent


st.set_page_config(page_title="Clearframe | Data agent", page_icon="CF", layout="wide")
st.title("Clearframe")
st.caption("A Nemotron-powered data cleaning agent")

with st.sidebar:
    st.subheader("Token Factory")
    api_key = os.getenv("NEBIUS_API_KEY", "")
    if not api_key:
        api_key = st.text_input("API key", type="password", help="Stored for this session only.")
    model = st.text_input("Model ID", value=os.getenv("NEBIUS_MODEL", DEFAULT_MODEL))
    st.caption(f"API: {BASE_URL}")
    if st.button("Test connection", disabled=not api_key, use_container_width=True):
        try:
            with st.spinner("Sending a test prompt..."):
                reply = DataAgent(api_key, model=model).smoke_test()
            st.success(reply or "The model returned an empty reply.")
        except Exception as error:
            st.error(f"Connection failed: {error}")
    st.link_button("Open model catalog", "https://tokenfactory.nebius.com/endpoints?modality=text2text", use_container_width=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    try:
        original_frame = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        st.stop()

    st.subheader(uploaded_file.name)
    st.dataframe(original_frame.head(100), use_container_width=True)
    task = st.text_area(
        "Data task",
        value="Inspect this dataset, clean obvious missing values and duplicate rows without discarding useful records, then summarize the changes.",
        height=100,
    )
    if st.button("Run agent", type="primary", disabled=not api_key or not task):
        agent = DataAgent(api_key, model=model)
        steps = st.status("Agent working", expanded=True)

        def show_step(step):
            steps.write(f"Round {step['round']}: `{step['tool']}` {step['status']}")
            if step["status"] == "error":
                steps.code(step["detail"])

        try:
            report, cleaned_frame, chart_path = agent.run(original_frame, task, on_step=show_step)
            steps.update(label="Agent finished", state="complete")
            st.session_state["cleaned_frame"] = cleaned_frame
            st.session_state["report"] = report
            st.session_state["chart_path"] = chart_path
        except Exception as error:
            steps.update(label="Agent stopped", state="error")
            st.error(str(error))

if "cleaned_frame" in st.session_state:
    st.subheader("Cleaned data")
    st.dataframe(st.session_state["cleaned_frame"].head(100), use_container_width=True)
    st.download_button(
        "Download cleaned CSV",
        data=st.session_state["cleaned_frame"].to_csv(index=False).encode("utf-8"),
        file_name="cleaned.csv",
        mime="text/csv",
    )
    st.subheader("Agent report")
    st.markdown(st.session_state["report"])
    chart_path = st.session_state.get("chart_path")
    if chart_path and os.path.exists(chart_path):
        st.image(chart_path, use_container_width=True)
    st.download_button(
        "Download report",
        data=st.session_state["report"].encode("utf-8"),
        file_name="report.md",
        mime="text/markdown",
    )