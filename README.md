# Ollama Chat Interface

A Streamlit-based chat interface for interacting with Ollama models, featuring a clean UI and model management capabilities.

## Features

- 🤖 Support for multiple Ollama models
- 💬 Interactive chat interface
- 🖼️ Image upload and processing capability
- 📊 Model information display
- 💾 Chat history with MongoDB storage
- 🔄 Response caching
- 🎨 Customizable UI elements

## Prerequisites

- Python 3.11+
- MongoDB
- Ollama installed and running locally
- Required Python packages (see requirements.txt)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ollama-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
MONGODB_URI=your_mongodb_connection_string
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Select a model from the sidebar
2. Configure model parameters (temperature, streaming)
3. Start chatting with the model
4. Optionally upload images for visual analysis
5. View chat history and model information

## Project Structure

```
├── .streamlit/          # Streamlit configuration
├── db_utils.py         # Database utilities
├── main.py            # Main application
├── utils.py           # Helper functions
├── styles.css         # Custom styling
└── requirements.txt   # Python dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License - feel free to use and modify as needed.
