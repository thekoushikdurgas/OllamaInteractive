import streamlit as st
from utils import get_ollama_response, format_message, get_available_models, get_model_details
from db_utils import save_message, get_chat_history, clear_chat_history
import os
from typing import Union, Generator

# Page configuration
st.set_page_config(
    page_title="Chat with Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = get_chat_history()
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "chat"

# Sidebar
with st.sidebar:
    st.title("Chat Settings")

    # New Chat Button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    # Model Selection
    st.markdown("### Available Models")
    available_models = get_available_models()

    if not available_models:
        st.warning("⚠️ Loading models... Please check Ollama connection")
        model_options = ["default"]
    else:
        model_options = [model["name"] for model in available_models]

    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=0,
        key="model_select"
    )

    # Model Information
    if selected_model and available_models:
        model_info = next((m for m in available_models if m["name"] == selected_model), None)
        if model_info:
            st.markdown("### Model Information")
            if "size_mb" in model_info:
                st.metric("Model Size", f"{model_info['size_mb']:.1f} MB")

            with st.expander("Technical Details"):
                if "details" in model_info:
                    details = model_info["details"]
                    st.markdown(f"**Format:** {details.get('format', 'Unknown')}")
                    st.markdown(f"**Family:** {details.get('family', 'Unknown')}")
                    st.markdown(f"**Parameters:** {details.get('parameter_size', 'Unknown')}")
                    st.markdown(f"**Quantization:** {details.get('quantization_level', 'None')}")

    # Advanced Settings
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        stream_enabled = st.toggle("Enable Streaming", value=True)

# Main chat interface
st.markdown("<div class='chat-header'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2,4,2])
with col1:
    mode = st.radio(
        "Mode",
        ["Chat", "Generate", "Code"],
        horizontal=True,
        key="chat_mode",
        label_visibility="collapsed"
    )
    st.session_state.current_mode = mode.lower()
st.markdown("</div>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    role_style = "user" if message["role"] == "user" else "assistant"
    st.markdown(f"<div class='message {role_style}-message'>{message['content']}</div>", unsafe_allow_html=True)

# Chat input
with st.container():
    input_placeholder = ("What's on your mind?" if st.session_state.current_mode == "chat" 
                        else "Enter prompt for generation..." if st.session_state.current_mode == "generate"
                        else "Enter code prompt...")

    user_input = st.text_area(
        "Message",
        key="user_input",
        placeholder=input_placeholder,
        label_visibility="collapsed",
        height=100
    )

    col1, col2 = st.columns([6,1])
    with col2:
        send_button = st.button("Send", use_container_width=True)

    if send_button and user_input:
        # Add user message
        save_message(user_input, "user", selected_model)
        st.session_state.messages = get_chat_history()

        # Get AI response
        with st.spinner("AI is thinking..."):
            response = get_ollama_response(
                user_input,
                selected_model,
                stream=stream_enabled,
                temperature=temperature
            )

            if isinstance(response, str):
                save_message(response, "assistant", selected_model)
                st.session_state.messages = get_chat_history()
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666666; padding: 1rem;'>
        <p>Current Model: {selected_model}</p>
        <p>Temperature: {temperature}</p>
        <p>Streaming: {'Enabled' if stream_enabled else 'Disabled'}</p>
    </div>
    """,
    unsafe_allow_html=True
)