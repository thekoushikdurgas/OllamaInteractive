import streamlit as st
import asyncio
from utils import get_ollama_response, get_ollama_response_async, format_message, get_available_models, get_model_details
from db_utils import save_message, get_chat_history, clear_chat_history
import os
from typing import Union, Generator

# Page configuration
st.set_page_config(
    page_title="Chat with Ollama",
    page_icon="🤖",
    layout="wide"
)

# Load custom CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize chat history from MongoDB
if "messages" not in st.session_state:
    messages = get_chat_history()
    st.session_state.messages = messages if messages is not None else []

# Model selection section
st.sidebar.title("Chat Settings")

# Model selection with description
st.sidebar.markdown("### Available Models")
st.sidebar.markdown("Choose a model to chat with:")

# Get available models with details
available_models = get_available_models()

# Create model selection options with loading indicator
if not available_models:
    st.sidebar.warning("⚠️ Loading models... Please make sure Ollama is running.")
    model_options = {"default": {"name": "default"}}
else:
    model_options = {model_info['name']: model_info for model_info in available_models}

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    index=0 if "llama2" in model_options else 0,
    help="Select the AI model you want to chat with"
)

# Show detailed model information
if selected_model:
    model_info = model_options[selected_model]
    st.sidebar.markdown("### Model Information")

    # Display basic info with improved formatting
    if isinstance(model_info, dict) and 'size_mb' in model_info:
        try:
            size_mb = float(model_info['size_mb'])
            size_display = f"{size_mb:.1f} MB"
            st.sidebar.metric("Model Size", size_display)
        except (ValueError, TypeError):
            st.sidebar.text(f"Size: {model_info.get('size_mb', 'Unknown')} MB")

    # Display detailed information if available
    if isinstance(model_info, dict) and model_info.get('details'):
        details = model_info['details']
        with st.sidebar.expander("Technical Details", expanded=False):
            specs = {
                "Format": str(details.get('format', 'Unknown')),
                "Family": str(details.get('family', 'Unknown')),
                "Parameters": str(details.get('parameter_size', 'Unknown')),
                "Quantization": str(details.get('quantization_level', 'None'))
            }
            for key, value in specs.items():
                st.markdown(f"**{key}:** {value}")

    # Add model capabilities hint with icons
    model_hints = {
        "llama": ("💡 General text generation and conversation", "#28a745"),
        "codellama": ("💻 Code generation and technical discussions", "#0056b3"),
        "mistral": ("⚖️ Balanced performance for various tasks", "#6f42c1"),
        "neural-chat": ("🗣️ Optimized for natural conversations", "#e83e8c")
    }

    for key, (hint, color) in model_hints.items():
        if key in selected_model.lower():
            st.sidebar.markdown(
                f"""<div style='padding: 10px; background-color: {color}20; 
                border-left: 3px solid {color}; margin: 10px 0;'>{hint}</div>""",
                unsafe_allow_html=True
            )

# Use selected_model instead of model variable
model = selected_model

# Advanced settings expander with improved UI
with st.sidebar.expander("Advanced Settings"):
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more creative but less focused"
        )
    with col2:
        use_streaming = st.checkbox(
            "Enable Streaming",
            value=True,
            help="Show responses as they are generated"
        )
        use_generate = st.checkbox(
            "Use Generate Mode",
            value=False,
            help="Use generation instead of chat mode"
        )

# Main chat interface
st.title("Chat with Ollama 🤖")

# Display status indicator
if not available_models:
    st.error("⚠️ Unable to connect to Ollama server. Please check if it's running.")

# Display chat messages from MongoDB with improved styling
for message in st.session_state.messages:
    st.markdown(
        format_message(message['content'], message['role']),
        unsafe_allow_html=True
    )

# Chat input with improved layout
with st.container():
    # Image upload with preview
    uploaded_file = st.file_uploader(
        "Upload an image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload an image to discuss with the AI"
    )

    # Save uploaded image
    image_path = None
    if uploaded_file:
        # Create images directory if it doesn't exist
        os.makedirs("uploaded_images", exist_ok=True)
        image_path = f"uploaded_images/{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded image in a card-like container
        st.markdown(
            f"""<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>
            <p style='margin-bottom: 5px; color: #666;'>Uploaded Image:</p>
            <img src='data:image/png;base64,{uploaded_file.getvalue().hex()}' 
            style='max-width: 300px; border-radius: 5px;'></div>""",
            unsafe_allow_html=True
        )

    # Chat input with send button
    col1, col2 = st.columns([5,1])
    with col1:
        user_input = st.text_input(
            "Type your message",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Type your message here..."
        )
    with col2:
        send_button = st.button("Send", use_container_width=True)

    if send_button and (user_input or image_path):
        # Save and display user message
        save_message(user_input, "user", model)
        st.session_state.messages = get_chat_history()

        # Response container with loading indicator
        response_container = st.empty()

        if use_streaming:
            # Stream response with progress bar
            full_response = ""
            with st.spinner("AI is thinking..."):
                async for response_chunk in get_ollama_response_async(
                    user_input,
                    model,
                    stream=True,
                    temperature=temperature,
                    image_path=image_path,
                    use_generate=use_generate
                ):
                    if isinstance(response_chunk, str):
                        full_response += response_chunk
                        response_container.markdown(
                            format_message(full_response, "assistant"),
                            unsafe_allow_html=True
                        )

            # Save full response
            save_message(full_response, "assistant", model)
        else:
            # Show loading spinner while getting response
            with st.spinner("AI is thinking..."):
                response = get_ollama_response(
                    user_input,
                    model,
                    temperature=temperature,
                    image_path=image_path
                )
                # Save and display assistant response
                if isinstance(response, str):
                    save_message(response, "assistant", model)
                    response_container.markdown(
                        format_message(response, "assistant"),
                        unsafe_allow_html=True
                    )

        st.session_state.messages = get_chat_history()
        st.session_state["user_input"] = "" #Corrected this line
        # Remove uploaded image
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        # Rerun to update chat display
        st.experimental_rerun()

# Clear chat button with confirmation in danger zone
with st.sidebar.expander("Danger Zone", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", type="primary"):
            st.session_state.confirm_clear = True
    with col2:
        if st.session_state.get("confirm_clear", False):
            if st.button("⚠️ Confirm", type="secondary"):
                clear_chat_history()
                st.session_state.messages = []
                st.session_state.confirm_clear = False
                st.experimental_rerun()

# Footer with model info and status
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666666; padding: 1rem;'>
        <p>Current Model: {model}</p>
        <p>Temperature: {temperature}</p>
        <p>Streaming: {'Enabled' if use_streaming else 'Disabled'}</p>
        <p>Built with ❤️ using Streamlit and Ollama</p>
    </div>
    """,
    unsafe_allow_html=True
)