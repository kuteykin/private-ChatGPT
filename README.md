# ChatGPT Clone - Streamlit Application

A unified Streamlit-based chat interface supporting multiple LLM providers (OpenAI, Anthropic, DeepSeek) with advanced features including file uploads, web search, and vision capabilities.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Data Flow](#data-flow)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Features

- **Multi-Provider Support**: OpenAI GPT models, Anthropic Claude models, and DeepSeek-R1 via Replicate
- **Multiple File Uploads**:
  - Upload multiple files at once for multimodal processing
  - PDF document processing (multiple PDFs supported)
  - Image analysis with vision models (multiple images supported)
  - Text file support (multiple text files supported)
  - Mix and match different file types in a single query
- **Web Search**:
  - Native integration for Claude models
  - Tavily API integration for OpenAI models
- **Multi-Turn Conversations**: Full conversation history with context preservation
- **Single-File Architecture**: Entire application in `chatgpt_streamlit.py`

## Architecture Overview

### Single-File Design

The application is intentionally designed as a monolithic single-file application (`chatgpt_streamlit.py`). All functionality including UI, API integration, file handling, and session management is contained in one Python file for simplicity and maintainability.

### Dual API Pattern

The application uses two different approaches for making LLM API calls:

#### 1. LangChain Wrappers (Default Path)
Used for basic chat interactions without file attachments or web search.

**When used:**
- Simple text-based conversations
- No file uploads
- Web search disabled

**Key functions:**
- `select_model()` - Initializes LangChain chat models (chatgpt_streamlit.py:139)
- `get_answer()` - Routes to LangChain when conditions are met (chatgpt_streamlit.py:422)

**Message flow:**
```
User Input → LangChain ChatModel → Response
```

#### 2. Native SDKs (Advanced Features Path)
Uses provider-specific SDKs (Anthropic/OpenAI) for advanced features.

**When used:**
- Web search is enabled (Claude only via native API)
- Files are uploaded (PDFs, images)
- Direct API features needed

**Key functions:**
- `get_answer_anthropic_native()` - Native Anthropic API with tool support (chatgpt_streamlit.py:204)
- `get_answer_openai_native()` - Native OpenAI API with Files API (chatgpt_streamlit.py:336)

**Message flow:**
```
User Input + Files/Tools → Native SDK → Tool Use Loop (if needed) → Response
```

### Model Configuration

Models are defined in dictionary mappings at the top of the file:

```python
ANTHROPIC_MODELS = {
    "Claude-4.1-Opus": "claude-opus-4-1-20250805",
    "Claude-4.5-Sonnet": "claude-sonnet-4-5",
    # ... more models
}

OPENAI_MODELS = {
    "OpenAI-GPT-5": "gpt-5",
    "OpenAI-GPT-4.1": "gpt-4.1-2025-04-14",
    "OpenAI-Reasoning-o4-mini": "o4-mini",
    # ... more models
}
```

Adding a new model requires:
1. Adding entry to appropriate dictionary
2. Updating radio button options in `select_model()`
3. Adding initialization logic with model-specific parameters

## Data Flow

### 1. User Input Flow

```
┌─────────────────┐
│  User Message   │
│  (Text Input)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ File Upload?    │◄──── Optional: PDF, Image, or Text file
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Session State   │
│   - messages    │
│   - file data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  get_answer()   │
│   (Router)      │
└────────┬────────┘
         │
         ├─────────────────┬─────────────────┐
         │                 │                 │
    ▼                 ▼                 ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ LangChain│    │Anthropic │    │  OpenAI  │
│  Wrapper │    │  Native  │    │  Native  │
└──────────┘    └──────────┘    └──────────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                          │
                          ▼
                  ┌─────────────┐
                  │  Response   │
                  └─────────────┘
```

### 2. File Upload Flow

#### OpenAI File Handling
```
File Upload
    │
    ▼
┌─────────────────────────┐
│ upload_file_to_openai() │
│  (Files API)            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Wait for processing     │
│ (status polling)        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Store file_id in        │
│ session_state           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Pass file_id to API     │
│ in message attachments  │
└─────────────────────────┘
```

#### Anthropic File Handling
```
File Upload
    │
    ├─── PDF ─────┐
    │             ▼
    │   ┌──────────────────┐
    │   │ extract_text()   │
    │   │ (PyMuPDF)        │
    │   └────────┬─────────┘
    │            │
    │            ▼
    │   ┌──────────────────┐
    │   │ Base64 encode    │
    │   └────────┬─────────┘
    │            │
    │            ▼
    │   ┌──────────────────┐
    │   │ Document content │
    │   │ block in message │
    │   └──────────────────┘
    │
    └─── Image ───┐
                  ▼
        ┌──────────────────┐
        │ Base64 encode    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Image content    │
        │ block in message │
        └──────────────────┘
```

### 3. Web Search Flow

#### Claude Web Search (Native API)
```
User Query + Web Search Enabled
            │
            ▼
┌─────────────────────────────┐
│ get_answer_anthropic_native │
│ with web_search_20250305    │
└──────────────┬──────────────┘
               │
               ▼
        ┌──────────────┐
        │  Tool Use    │◄────┐
        │    Loop      │     │
        └──────┬───────┘     │
               │             │
               ├─ Tool Call? ┤
               │             │
               ▼             │
        ┌──────────────┐    │
        │ Web Search   │────┘
        │   Result     │
        └──────────────┘
               │
               ▼
        ┌──────────────┐
        │ Final Text   │
        │   Response   │
        └──────────────┘
```

#### OpenAI Web Search (Tavily API)
```
User Query + Web Search Enabled
            │
            ▼
┌───────────────────────────┐
│ get_answer_openai_native  │
│ with web_search function  │
└──────────────┬────────────┘
               │
               ▼
        ┌──────────────┐
        │  Tool Use    │◄────┐
        │    Loop      │     │
        └──────┬───────┘     │
               │             │
               ├─ Tool Call? ┤
               │             │
               ▼             │
        ┌──────────────┐    │
        │ web_search() │    │
        │  (Tavily)    │────┘
        └──────────────┘
               │
               ▼
        ┌──────────────┐
        │ Final Text   │
        │   Response   │
        └──────────────┘
```

### 4. Session State Management

Streamlit session state stores conversation context and file data:

```python
st.session_state = {
    # Conversation history (LangChain format)
    "messages": [SystemMessage, HumanMessage, AIMessage, ...],

    # Model selection
    "selected_model": "Claude 4.5 Sonnet",

    # Feature flags
    "enable_web_search": False,

    # OpenAI file references (lists for multiple files)
    "last_file_ids": ["file-abc123", "file-def456"],

    # Anthropic PDF data (lists for multiple PDFs)
    "last_pdf_contents": ["base64_encoded_content1", "base64_encoded_content2"],
    "last_pdf_filenames": ["document1.pdf", "document2.pdf"],

    # Image data for both providers (lists for multiple images)
    "last_image_contents": ["base64_encoded_image1", "base64_encoded_image2"],
    "last_image_mime_types": ["image/png", "image/jpeg"],
    "last_images": [
        {
            "name": "screenshot1.png",
            "data": "data:image/png;base64,..."
        },
        {
            "name": "screenshot2.jpg",
            "data": "data:image/jpeg;base64,..."
        }
    ]
}
```

### 5. Tool Use Loop (Web Search)

Both providers implement a tool use loop for web search:

```
┌─────────────────┐
│ Initial Request │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  API Call       │
└────────┬────────┘
         │
         ▼
    ┌────────────┐
    │ Response   │
    └─────┬──────┘
          │
     ┌────┴────┐
     │         │
     ▼         ▼
 Tool Use?   Final Text
     │           │
     │           ▼
     │      ┌─────────┐
     │      │ Return  │
     │      └─────────┘
     │
     ▼
┌─────────────────┐
│ Execute Tool    │
│ (Web Search)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Append Tool     │
│ Result          │
└────────┬────────┘
         │
         └──────► (Loop back to API Call)
                   Max 10 iterations
```

## Installation

### Prerequisites

- Python ^3.11
- Poetry (recommended) or pip

### Setup with Poetry

```bash
# Clone the repository
git clone <repository-url>
cd ChatGPT_clone

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Setup with pip

```bash
# Clone the repository
git clone <repository-url>
cd ChatGPT_clone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit langchain-openai langchain-anthropic langchain-community openai anthropic replicate python-dotenv
```

### Environment Configuration

Create a `.chat-env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
REPLICATE_API_TOKEN=your_replicate_token_here  # For DeepSeek-R1
TAVILY_API_KEY=your_tavily_key_here  # For web search with OpenAI models
```

## Usage

### Running the Application

```bash
# Run with default port (8501)
streamlit run chatgpt_streamlit.py

# Run with custom port
streamlit run chatgpt_streamlit.py --server.port 8503
```

The application will open in your default browser at `http://localhost:8501` (or your custom port).

### Using Docker

```bash
# Build the Docker image
docker build -t chatgpt-clone .

# Run the container (exposes port 8503)
docker run -p 8503:8503 chatgpt-clone
```

Access the application at `http://localhost:8503`

### Features Usage

#### Basic Chat
1. Select a model from the sidebar
2. Type your message in the input box
3. Press Enter or click outside the input

#### File Upload
1. Click "Upload PDF, images, or text files" in the sidebar
2. Select one or multiple files (hold Ctrl/Cmd to select multiple)
3. Supported formats: PDF, PNG, JPG, JPEG, GIF, WEBP, TXT, MD, CSV
4. The file icons (📎) will appear in your message showing which files were attached
5. Ask questions about the uploaded content
6. You can mix different file types in a single upload (e.g., 2 PDFs + 3 images)

#### Web Search
1. Enable "Enable Web Search" checkbox in the sidebar
2. For Claude models: Uses native Anthropic web search
3. For OpenAI models: Requires Tavily API key
4. Ask questions requiring current information

#### Image Analysis
1. Upload one or multiple image files
2. Compatible with vision models (GPT-4, Claude Sonnet/Opus)
3. Ask questions about image content
4. Compare multiple images in a single query

## Configuration

### Adding New Models

To add a new model, edit `chatgpt_streamlit.py`:

1. **Add to model dictionary** (line 21 or 27):
```python
ANTHROPIC_MODELS = {
    "New Model Name": "api-model-identifier",
    # ... existing models
}
```

2. **Update select_model() function** (line 139):
```python
if selected_model == "New Model Name":
    st.session_state['chat'] = ChatAnthropic(
        model=ANTHROPIC_MODELS[selected_model],
        temperature=0.7,
        # ... model-specific parameters
    )
```

3. **Add to radio button options**:
```python
selected_model = st.radio(
    "Select Model",
    ["New Model Name"] + list(ANTHROPIC_MODELS.keys())
)
```

### Customizing Temperature and Parameters

Model parameters are set in the `select_model()` function. Adjust temperature, max_tokens, and other parameters as needed:

```python
st.session_state['chat'] = ChatAnthropic(
    model=ANTHROPIC_MODELS[selected_model],
    temperature=0.7,  # Adjust creativity (0.0 - 1.0)
    max_tokens=4096,  # Maximum response length
)
```

## Project Structure

```
ChatGPT_clone/
├── chatgpt_streamlit.py    # Main application file (entire app)
├── pyproject.toml           # Poetry dependencies
├── Dockerfile               # Docker configuration
├── .chat-env                # Environment variables (create this)
└── README.md                # This file
```

## Key Functions Reference

| Function | Location | Purpose |
|----------|----------|---------|
| `select_model()` | chatgpt_streamlit.py:139 | Model selection and initialization |
| `get_answer()` | chatgpt_streamlit.py:422 | Main routing logic for API calls |
| `get_answer_anthropic_native()` | chatgpt_streamlit.py:204 | Native Anthropic API with tools |
| `get_answer_openai_native()` | chatgpt_streamlit.py:336 | Native OpenAI API with files |
| `upload_file_to_openai()` | chatgpt_streamlit.py:77 | OpenAI Files API integration |
| `extract_text_from_pdf()` | chatgpt_streamlit.py:40 | PDF text extraction for Anthropic |
| `web_search()` | chatgpt_streamlit.py:59 | Tavily web search for OpenAI |

## Dependencies

Core dependencies (see `pyproject.toml`):

- `streamlit` - Web UI framework
- `langchain-openai` - OpenAI LangChain integration
- `langchain-anthropic` - Anthropic LangChain integration
- `langchain-community` - Community LangChain components
- `openai` - Native OpenAI SDK (for Files API)
- `anthropic` - Native Anthropic SDK (for web search & tools)
- `replicate` - DeepSeek-R1 via Replicate
- `python-dotenv` - Environment variable management

## Troubleshooting

### "API key not found" errors
- Ensure `.chat-env` file exists in project root
- Verify all required API keys are set
- Check that keys are valid and have appropriate permissions

### File upload not working
- For OpenAI: Check file size limits (max 512 MB)
- For PDFs: Ensure PyMuPDF is installed
- Verify file permissions

### Web search not working
- Claude models: Verify Anthropic API key has tool use enabled
- OpenAI models: Verify TAVILY_API_KEY is set in `.chat-env`
- Check internet connectivity

### Model not responding
- Check API key validity
- Verify model identifier is correct
- Check for rate limiting or quota issues with provider

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
