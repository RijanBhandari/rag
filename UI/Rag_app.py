import streamlit as st
from rag_backend import embbed_and_store, generate_response, extract_text_from_chunk

st.title("🤖 Local Chatbot")


# Initialize chat history and index storage in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

# Static greeting (doesn't append to history so it won't duplicate)
with st.chat_message("assistant"):
    st.markdown("Hi, How can I help you today?")

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input block
if prompt := st.chat_input(
    placeholder="Type a message or upload a file...",
    accept_file=True,
    file_type=["pdf"],
):
    uploaded_files = prompt.files
    query = prompt.text

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
        if uploaded_files:
            for file in uploaded_files:
                st.caption(f"📎 Attached: {file.name}")

    # Now append to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Process files if uploaded and update the stored index
    if uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.index = embbed_and_store(uploaded_files)

    # Generate and stream assistant response
    # Pass history BEFORE the current message (already appended above)
    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        response_stream = generate_response(st.session_state.index, query, history=history)
        full_response = st.write_stream(
            extract_text_from_chunk(chunk) for chunk in response_stream
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})