import requests
import streamlit as st
import json
import sys
from pathlib import Path



# Resolve project root (the inner finsight-financial-rag-system-main dir)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent/"finsight-financial-rag-system"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
    from app.utils.config import get_settings

settings = get_settings()

st.set_page_config(page_title="FinSight", layout="wide")
st.title("FinSight")
st.caption("Financial document intelligence with explainable RAG answers.")

api_base_url = st.sidebar.text_input("API base URL", settings.streamlit_api_base_url)

with st.sidebar:
    if st.button("Check API"):
        try:
            response = requests.get(f"{api_base_url}/api/v1/health", timeout=10)
            response.raise_for_status()
            st.success(f"API is {response.json()['status']}")
        except requests.RequestException as exc:
            st.error(f"API check failed: {exc}")

uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
if uploaded_file is not None and st.button("Upload PDF"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    try:
        response = requests.post(
            f"{api_base_url}/api/v1/documents/upload",
            files=files,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        st.success(payload["message"])
        st.write(
            {
                "filename": payload["filename"],
                "total_pages": payload["total_pages"],
                "extracted_pages": payload["extracted_pages"],
                "total_chunks": payload["total_chunks"],
            }
        )
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")

question = st.text_area("Ask a question about uploaded documents", height=120)
enable_evaluation = st.checkbox("Enable evaluation metrics", value=False)

if st.button("Ask") and question.strip():
    try:
        if enable_evaluation:
            endpoint = f"{api_base_url}/api/v1/evaluate"
            payload_data = {"question": question, "top_k": settings.top_k}
        else:
            endpoint = f"{api_base_url}/api/v1/qa"
            payload_data = {"question": question, "top_k": settings.top_k}

        response = requests.post(
            endpoint,
            json=payload_data,
            timeout=60,
        )
        if response.status_code == 501:
            st.info(response.json()["detail"])
        else:
            response.raise_for_status()
            payload = response.json()
            st.subheader("Answer")
            st.write(payload["answer"])
            st.caption(
                f"Model: {payload.get('model', 'N/A')} | "
                f"Response chars: {payload.get('response_chars', 'N/A')} | "
                f"Latency: {payload.get('response_latency_ms', 'N/A')}ms"
            )

            st.subheader("Citations")
            if payload["citations"]:
                for citation in payload["citations"]:
                    page = citation["page_number"] or "unknown"
                    section = citation.get("section_title") or "Unknown section"
                    st.markdown(f"- **{citation['source_filename']}**, page {page}, {section}")
            else:
                st.info("No citations were available for this answer.")

            st.subheader("Retrieved evidence")
            for index, result in enumerate(payload.get("retrieved_chunks", []), start=1):
                label = (
                    f"[{index}] {result['source_filename']} p.{result['page_number']} "
                    f"score {result['similarity_score']:.3f}"
                )
                with st.expander(label):
                    if result.get("section_title"):
                        st.caption(result["section_title"])
                    st.write(result["text"])

            # Evaluation section (if enabled)
            if enable_evaluation and "metrics" in payload:
                st.divider()
                st.subheader("Evaluation Results")

                # Metrics display
                col1, col2, col3, col4 = st.columns(4)
                metrics = payload["metrics"]
                with col1:
                    st.metric("Faithfulness", f"{metrics['faithfulness']:.2%}")
                with col2:
                    st.metric("Answer Relevancy", f"{metrics['answer_relevancy']:.2%}")
                with col3:
                    st.metric("Context Precision", f"{metrics['context_precision']:.2%}")
                with col4:
                    st.metric("Context Recall", f"{metrics['context_recall']:.2%}")

                # Retrieval metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieval Precision", f"{metrics['retrieval_precision']:.2%}")
                with col2:
                    st.metric("NDCG@5", f"{metrics['retrieval_ndcg']:.2%}")
                with col3:
                    st.metric("Chunks Retrieved", payload.get("retrieved_count", 0))

                # Hallucination analysis
                hallucinations = payload.get("hallucination_warnings", [])
                hallucination_rate = payload.get("hallucination_rate", 0.0)

                st.metric("Hallucination Rate", f"{hallucination_rate:.2%}")

                if hallucinations:
                    st.warning(f"⚠️ {len(hallucinations)} potential hallucinations detected:")
                    for h in hallucinations:
                        confidence_level = "🔴 HIGH" if h["confidence"] > 0.8 else "🟡 MEDIUM" if h["confidence"] > 0.6 else "🟢 LOW"
                        st.markdown(f"**{confidence_level}** (confidence: {h['confidence']:.2%})")
                        st.markdown(f"- *{h['sentence'][:100]}...*")
                        st.caption(h["explanation"])
                else:
                    st.success("✅ No hallucinations detected")

                # Retrieval stats
                if payload.get("retrieval_stats"):
                    with st.expander("Retrieval Statistics"):
                        stats = payload["retrieval_stats"]
                        st.json(stats)

                # Export option
                if st.button("Export evaluation as JSON"):
                    st.download_button(
                        label="Download evaluation.json",
                        data=json.dumps(payload, indent=2),
                        file_name="evaluation.json",
                        mime="application/json",
                    )

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
