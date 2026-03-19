import streamlit as st
from retriever import load_chunks, search
from pruner import prune_and_answer

st.set_page_config(
    page_title="Science Tutor",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}
section[data-testid="stSidebar"] {
    background-color: #f7f7f8;
    border-right: 1px solid #e5e5e5;
}
.chat-question {
    background: #1a73e8;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 10px 0 4px auto;
    font-size: 15px;
    max-width: 75%;
    float: right;
    clear: both;
}
.chat-answer {
    background: #f0f0f0;
    color: #1a1a1a;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    margin: 4px 0 10px 0;
    font-size: 15px;
    line-height: 1.8;
    max-width: 75%;
    float: left;
    clear: both;
}
.clearfix { clear: both; }
.tutor-label {
    font-size: 12px;
    color: #888;
    margin: 2px 0 2px 4px;
    clear: both;
    display: block;
}
.stTextInput input {
    border-radius: 25px !important;
    padding: 12px 20px !important;
    font-size: 15px !important;
    border: 1.5px solid #e0e0e0 !important;
    background: #fafafa !important;
}
.stTextInput input:focus {
    border-color: #1a73e8 !important;
    background: white !important;
}
.history-item {
    background: white;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 13px;
    color: #333;
    border: 1px solid #e5e5e5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "input_key" not in st.session_state:
    st.session_state["input_key"] = 0

@st.cache_resource
def load_chunks_cached():
    return load_chunks("data/science_class10.txt")

chunks = load_chunks_cached()

with st.sidebar:
    st.markdown("## Science Tutor")
    st.caption("Class 10 Science")
    st.divider()

    if st.button("New Chat", use_container_width=True, type="primary"):
        st.session_state["chat_history"] = []
        st.session_state["input_key"] += 1
        st.rerun()

    st.divider()

    st.markdown("#### Recent Questions")
    if st.session_state["chat_history"]:
        for i, chat in enumerate(reversed(st.session_state["chat_history"])):
            short = chat["question"][:40] + "..." if len(chat["question"]) > 40 else chat["question"]
            st.markdown(
                f'<div class="history-item">{short}</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("Your questions will appear here")

    st.divider()

    st.markdown("#### Try asking:")
    suggestions = [
        "What is photosynthesis?",
        "How does the human heart work?",
        "What is Ohm's law?",
        "How does refraction work?",
        "What is a chemical reaction?",
        "How does digestion work?",
        "What is the water cycle?",
        "What are acids and bases?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"s_{s}"):
            st.session_state["pending_question"] = s
            st.rerun()

    st.divider()
    st.caption("Class 10 Science Tutor")

st.markdown("## Science Tutor")
st.caption("Class 10 Science — Ask anything from your textbook")
st.divider()

for chat in st.session_state["chat_history"]:
    st.markdown(
        f'<div class="chat-question">{chat["question"]}</div>'
        f'<div class="clearfix"></div>',
        unsafe_allow_html=True
    )
    st.markdown('<span class="tutor-label">Tutor</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="chat-answer">{chat["answer"]}</div>'
        f'<div class="clearfix"></div>',
        unsafe_allow_html=True
    )

pending = st.session_state.pop("pending_question", "") if "pending_question" in st.session_state else ""

question = st.text_input(
    label="question",
    placeholder="Ask a question from Class 10 Science...",
    value=pending,
    key=f"q_{st.session_state['input_key']}",
    label_visibility="collapsed"
)

col1, col2 = st.columns([5, 1])
with col1:
    ask_button = st.button("Ask", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["input_key"] += 1
        st.rerun()

def get_real_answer(question):
    relevant = search(question, chunks, top=5)
    result = prune_and_answer(question, relevant)
    return result

if ask_button and question.strip():
    with st.spinner("Finding answer..."):
        result = get_real_answer(question)
    st.session_state["chat_history"].append({
        "question": question,
        "answer": result["answer"]
    })
    st.session_state["input_key"] += 1
    st.rerun()

elif ask_button and not question.strip():
    st.warning("Please type a question first!")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:12px;'>"
    "Class 10 Science Tutor"
    "</div>",
    unsafe_allow_html=True
)