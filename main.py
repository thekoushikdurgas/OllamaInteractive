import streamlit as st
from utils import get_ollama_response, format_message
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
st.sidebar.title("Chat Settings")

# Model selection
model = st.sidebar.selectbox(
    "Select Model",
    ["llama2", "mistral", "codellama"],
    index=0
)

# Temperature slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.1
)

# Main chat interface
st.title("Chat with Ollama 🤖")

# Display chat messages
for message in st.session_state.messages:
    st.markdown(
        format_message(message["content"], message["role"]),
        unsafe_allow_html=True
    )

# Chat input
with st.container():
    # Create two columns for input and button
    col1, col2 = st.columns([5,1])
    
    with col1:
        user_input = st.text_input("Type your message", key="user_input", label_visibility="collapsed")
    
    with col2:
        send_button = st.button("Send")

    if send_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show loading spinner while getting response
        with st.spinner("Thinking..."):
            # Get assistant response
            response = get_ollama_response(user_input, model)
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear input
        st.session_state.user_input = ""
        
        # Rerun to update chat display
        st.experimental_rerun()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666; padding: 1rem;'>
        Built with Streamlit and Ollama
    </div>
    """,
    unsafe_allow_html=True
)
