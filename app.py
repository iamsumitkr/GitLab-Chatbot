import streamlit as st
from rag_chain import get_qa_chain

st.set_page_config(page_title="GitLab Chatbot", layout="wide")

st.title("💬 GitLab Handbook Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa_chain = get_qa_chain()

query = st.chat_input("Ask a question about GitLab...")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"input": query})
        answer = result["answer"]
        sources = result["context"]

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("bot", answer))

# Display chat
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# Optional: show sources
if query:
    with st.expander("📚 Sources"):
        for doc in sources:
            st.write(doc.page_content[:300])
