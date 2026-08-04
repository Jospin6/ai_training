from __future__ import annotations

import json
import sys
from typing import Any
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.genai.multi_agent.pipelines.pipeline import run_research_pipeline


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _render_result(result: dict[str, Any]) -> None:
    report = result.get("report")
    feedback = result.get("feedback")

    left, right = st.columns(2)

    with left:
        st.subheader("Report")
        if report:
            st.markdown(_as_text(report))
        else:
            st.info("No report returned by the pipeline.")

    with right:
        st.subheader("Critic feedback")
        if feedback:
            st.markdown(_as_text(feedback))
        else:
            st.info("No critic feedback returned by the pipeline.")

    with st.expander("Search results", expanded=False):
        search_results = _as_text(result.get("search_results"))
        if search_results:
            st.code(search_results, language="text")
        else:
            st.info("No search results available.")

    with st.expander("Scraped content", expanded=False):
        scraped_content = _as_text(result.get("scraped_content"))
        if scraped_content:
            st.code(scraped_content, language="text")
        else:
            st.info("No scraped content available.")

    with st.expander("Raw pipeline output", expanded=False):
        st.json(result, expanded=False)


def main() -> None:
    st.set_page_config(
        page_title="Multi-agent research",
        page_icon="🔎",
        layout="wide",
    )

    st.title("Multi-agent research interface")
    st.caption(
        "Enter a topic, run `run_research_pipeline`, and inspect the output."
    )

    with st.sidebar:
        st.header("Pipeline")
        st.write("1. Search agent")
        st.write("2. Reader agent")
        st.write("3. Writer chain")
        st.write("4. Critic chain")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "last_topic" not in st.session_state:
        st.session_state.last_topic = ""

    if "last_error" not in st.session_state:
        st.session_state.last_error = ""

    with st.form("research_form", clear_on_submit=False):
        topic = st.text_input(
            "Topic",
            placeholder="Example: Impact of generative AI in education",
            key="topic_input",
        )
        submitted = st.form_submit_button("Run pipeline", use_container_width=True)

    if submitted:
        topic = topic.strip()

        if not topic:
            st.warning("Please enter a topic before running the pipeline.")
        else:
            st.session_state.last_error = ""

            with st.spinner("Running the research pipeline..."):
                try:
                    result = run_research_pipeline(topic)
                    st.session_state.last_topic = topic
                    st.session_state.last_result = result
                except Exception as error:
                    st.session_state.last_result = None
                    st.session_state.last_error = str(error)

    if st.session_state.last_error:
        st.error(f"Pipeline error: {st.session_state.last_error}")

    if st.session_state.last_result:
        st.success(f"Completed topic: {st.session_state.last_topic}")
        _render_result(st.session_state.last_result)
    elif not st.session_state.last_error:
        st.info("Submit a topic to launch the research pipeline.")


if __name__ == "__main__":
    main()
