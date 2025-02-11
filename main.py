import streamlit as st
from utils import get_ollama_response, format_message, get_available_models, get_model_details, encode_image
from db_utils import save_message, get_chat_history, clear_chat_history
import os

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
    st.session_state.messages = get_chat_history()

# Model selection section in main.py
st.sidebar.title("Chat Settings")

# Model selection with description
st.sidebar.markdown("### Select Model")
st.sidebar.markdown("Choose a model to chat with:")

# Get available models
available_models = get_available_models()

# Model selection
model = st.sidebar.selectbox(
    "Available Models",
    available_models,
    index=0 if "llama2" in available_models else 0,
    help="Select the AI model you want to chat with"
)

# Show model details in an organized way
if model:
    model_info = get_model_details(model)
    st.sidebar.markdown("### Model Information")

    # Display model details in a clean format
    for key, value in model_info.items():
        st.sidebar.markdown(f"**{key}:** {value}")

    # Add model capabilities hint
    if "llama" in model.lower():
        st.sidebar.info("💡 This model is optimized for general text generation and conversation.")
    elif "codellama" in model.lower():
        st.sidebar.info("💡 This model specializes in code generation and technical discussions.")
    elif "mistral" in model.lower():
        st.sidebar.info("💡 This model offers balanced performance for various tasks.")


# Advanced settings expander
with st.sidebar.expander("Advanced Settings"):
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make the output more creative but less focused"
    )

    use_streaming = st.checkbox(
        "Enable Streaming",
        value=True,
        help="Show responses as they are generated"
    )

# Main chat interface
st.title("Chat with Ollama 🤖")

# Display chat messages from MongoDB
for message in st.session_state.messages:
    st.markdown(
        format_message(message['content'], message['role']),
        unsafe_allow_html=True
    )

# Chat input
with st.container():
    # Image upload
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

        # Display uploaded image
        st.image(image_path, caption="Uploaded Image")

    col1, col2 = st.columns([5,1])

    with col1:
        user_input = st.text_input(
            "Type your message",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Type your message here..."
        )

    with col2:
        send_button = st.button("Send")

    if send_button and (user_input or image_path):
        # Save and display user message
        save_message(user_input, "user", model)
        st.session_state.messages = get_chat_history()

        # Response container
        response_container = st.empty()

        if use_streaming:
            # Stream response
            full_response = ""
            for response_chunk in get_ollama_response(
                user_input,
                model,
                stream=True,
                temperature=temperature,
                image_path=image_path
            ):
                full_response += response_chunk
                response_container.markdown(
                    format_message(full_response, "assistant"),
                    unsafe_allow_html=True
                )

            # Save full response
            save_message(full_response, "assistant", model)
        else:
            # Show loading spinner while getting response
            with st.spinner("Thinking..."):
                response = get_ollama_response(
                    user_input,
                    model,
                    temperature=temperature,
                    image_path=image_path
                )
                # Save and display assistant response
                save_message(response, "assistant", model)
                response_container.markdown(
                    format_message(response, "assistant"),
                    unsafe_allow_html=True
                )

        st.session_state.messages = get_chat_history()
        # Clear input
        st.session_state.user_input = ""
        # Remove uploaded image
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        # Rerun to update chat display
        st.experimental_rerun()

# Clear chat button with confirmation
with st.sidebar.expander("Danger Zone"):
    if st.button("Clear Chat History"):
        if st.button("⚠️ Confirm Clear Chat"):
            clear_chat_history()
            st.session_state.messages = []
            st.experimental_rerun()

# Footer with model info
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666666; padding: 1rem;'>
        <p>Current Model: {model}</p>
        <p>Temperature: {temperature}</p>
        <p>Streaming: {'Enabled' if use_streaming else 'Disabled'}</p>
        <p>Built with Streamlit and Ollama</p>
    </div>
    """,
    unsafe_allow_html=True
)