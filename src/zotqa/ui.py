"""Streamlit chat UI for zotqa."""

import streamlit as st

from zotqa.rag import QueryEngine, VectorIndex, get_default_index_dir


def init_engine() -> QueryEngine | None:
    """Initialize the query engine, returning None if no index exists."""
    index_dir = get_default_index_dir()
    index = VectorIndex(index_dir)
    if not index.load():
        return None
    return QueryEngine(index_dir=index_dir)


def main() -> None:
    st.set_page_config(page_title="ZotQA", page_icon="📚", layout="centered")
    st.title("📚 Zot QA")
    st.caption("Ask questions about your Zotero library")

    # Initialize engine
    if "engine" not in st.session_state:
        st.session_state.engine = init_engine()

    if st.session_state.engine is None:
        st.error("No index found. Run `zotqa index <corpus_dir>` first.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📎 Sources"):
                    for source in message["sources"]:
                        st.caption(source)

    # Chat input
    if prompt := st.chat_input("Ask about your papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching your library..."):
                result = st.session_state.engine.query(prompt)

            st.markdown(result.answer)

            # Show sources
            sources = []
            if result.used_chunks:
                with st.expander("📎 Sources"):
                    for chunk in result.used_chunks:
                        cite = chunk.cite_key or chunk.paper_id
                        source_line = f"[{cite} | {chunk.paper_id} | {chunk.section}] (score: {chunk.score:.3f})"
                        sources.append(source_line)
                        st.caption(source_line)

                # Token usage in smaller text
                st.caption(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")

        # Save assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
