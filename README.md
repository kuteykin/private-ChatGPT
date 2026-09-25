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
  - Claude models: native Anthropic web search (`web_search_20250305` tool)
  - OpenAI models: native OpenAI web search (`web_search_preview` tool via Responses API)
- **Reasoning Effort Control**: Low / Medium / High reasoning effort for GPT-5.4 and Claude models
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
- `select_model()` - Initializes LangChain chat models
- `get_answer()` - Routes to LangChain when conditions are met

**Message flow:**
```
User Input → LangChain ChatModel → Response
```

#### 2. Native SDKs (Advanced Features Path)
Uses provider-specific SDKs (Anthropic/OpenAI) for advanced features.

**When used:**
- Web search is enabled
- Files are uploaded (PDFs, images)
- Reasoning effort is set (GPT-5.4 always uses this path)

**Key functions:**
- `get_answer_anthropic_native()` - Native Anthropic API with tool support
- `get_answer_openai_native()` - Native OpenAI Responses API with reasoning and web search

**Message flow:**
```
User Input + Files/Tools → Native SDK → Tool Use Loop (if needed) → Response
```

### Model Configuration

Models are defined in dictionary mappings at the top of the file:

```python
ANTHROPIC_MODELS = {
    "Claude-Opus-5.5": "claude-opus-5-5",
    "Claude-Sonnet-5": "claude-sonnet-5",
}

OPENAI_MODELS = {
    "OpenAI-GPT-5.4": "gpt-5.4",
    "OpenAI-GPT-5.4-mini": "gpt-5.4-mini",
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

Both providers use native web search tools — no external search API required.

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

#### OpenAI Web Search (Native Responses API)
```
User Query + Web Search Enabled
            │
            ▼
┌───────────────────────────┐
│ get_answer_openai_native  │
│ with web_search_preview   │
└──────────────┬────────────┘
               │
               ▼
        ┌──────────────┐
        │  Responses   │
        │  API Output  │
        └──────┬───────┘
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
    "selected_model": "OpenAI-GPT-5.4",

    # Feature flags
    "enable_web_search": False,
    "reasoning_effort": "Low",

    # OpenAI file references (lists for multiple files)
    "last_file_ids": ["file-abc123", "file-def456"],

    # Anthropic PDF data (lists for multiple PDFs)
    "last_pdf_contents": ["base64_encoded_content1", "base64_encoded_content2"],
    "last_pdf_filenames": ["document1.pdf", "document2.pdf"],

    # Image data for both providers (lists for multiple images)
    "last_image_contents": ["base64_encoded_image1", "base64_encoded_image2"],
    "last_image_mime_types": ["image/png", "image/jpeg"],
    "last_images": [
        {"name": "screenshot1.png", "data": "data:image/png;base64,..."},
        {"name": "screenshot2.jpg", "data": "data:image/jpeg;base64,..."}
    ]
}
```

### 5. Tool Use Loop (Claude Web Search)

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

### Environment Configuration

Create a `.chat-env` file in the project root before running the app:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
REPLICATE_API_TOKEN=your_replicate_token_here  # For DeepSeek-R1
```

### Option A — Poetry (local development)

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run the app
streamlit run chatgpt_streamlit.py

# Or run without activating the shell
poetry run streamlit run chatgpt_streamlit.py
```

The app will open at `http://localhost:8501`.

To use a different port:

```bash
poetry run streamlit run chatgpt_streamlit.py --server.port 8503
```

### Option B — Docker

The Docker image uses a two-stage build: Poetry resolves and installs dependencies in a builder stage, then copies only the virtualenv into a slim runtime image.

**Build the image:**

```bash
docker build -t chatgpt-clone .
```

**Run the container:**

The Dockerfile copies `.chat-env` into the image at build time, so your API keys are baked in. This is convenient for local use — run with:

```bash
docker run -p 8503:8503 chatgpt-clone
```

Access the app at `http://localhost:8503`.

**Run without baking secrets into the image (recommended for shared environments):**

Remove the `COPY .chat-env ./` line from the Dockerfile, rebuild, then mount the file at runtime:

```bash
docker run -p 8503:8503 -v "$(pwd)/.chat-env:/app/.chat-env:ro" chatgpt-clone
```

**Rebuild after dependency changes:**

Whenever `pyproject.toml` or `poetry.lock` changes, rebuild the image:

```bash
docker build --no-cache -t chatgpt-clone .
docker run -p 8503:8503 chatgpt-clone
```

### Setup with pip (alternative)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install streamlit langchain-openai langchain-anthropic langchain-community \
    openai anthropic replicate python-dotenv
```

## Usage

### Features Usage

#### Basic Chat
1. Select a model from the sidebar
2. Type your message in the input box
3. Press Enter

#### File Upload
1. Click "Upload PDF, images, or text files"
2. Select one or multiple files (hold Ctrl/Cmd to select multiple)
3. Supported formats: PDF, PNG, JPG, JPEG, GIF, WEBP, TXT, MD, CSV
4. The file icons (📎) will appear in your message showing attached files
5. Ask questions about the uploaded content

#### Web Search
1. Enable "Enable Web Search" checkbox in the sidebar
2. Both Claude and OpenAI models use their respective native web search tools
3. Ask questions requiring current information

#### Reasoning Effort
1. Select reasoning effort from the sidebar: None / Low / Medium / High
2. Higher effort improves response quality but uses more tokens and takes longer
3. For GPT-5.4 with Medium/High + Web Search, the model needs significant token budget — this is handled automatically

#### Image Analysis
1. Upload one or multiple image files
2. Ask questions about image content

## Configuration

### Adding New Models

To add a new model, edit `chatgpt_streamlit.py`:

1. **Add to model dictionary** (line ~21 or ~27):
```python
ANTHROPIC_MODELS = {
    "New Model Name": "api-model-identifier",
    # ... existing models
}
```

2. **Update the radio button options** in `select_model()`:
```python
ai_model = st.sidebar.radio(
    "Choose LLM:",
    ("New Model Name", "OpenAI-GPT-5.4", ...)
)
```

3. **Add initialization logic** inside `select_model()` if the model needs special handling.

### Customizing Model Parameters

Model parameters are set in the `select_model()` function:

```python
return ChatAnthropic(
    temperature=0.0,
    max_tokens=4096,
    model=model_name,
)
```

## Project Structure

```
ChatGPT_clone/
├── chatgpt_streamlit.py    # Main application file (entire app)
├── pyproject.toml           # Poetry dependencies
├── poetry.lock              # Locked dependency versions
├── Dockerfile               # Two-stage Docker build
├── .chat-env                # API keys (create this, never commit)
└── README.md                # This file
```

## Key Functions Reference

| Function | Purpose |
|----------|---------|
| `select_model()` | Model selection and initialization |
| `get_answer()` | Main routing logic — decides which API path to use |
| `get_answer_anthropic_native()` | Native Anthropic API with web search and file support |
| `get_answer_openai_native()` | Native OpenAI Responses API with reasoning and web search |
| `upload_file_to_openai()` | Uploads file to OpenAI Files API, waits for processing |
| `process_uploaded_files()` | Handles multiple file uploads, detects type per file |
| `convert_messages_to_anthropic()` | Converts LangChain messages to Anthropic format with attachments |
| `convert_messages_to_openai()` | Converts LangChain messages to OpenAI format with attachments |

## Dependencies

Core dependencies (see `pyproject.toml`):

- `streamlit` - Web UI framework
- `langchain-openai` - OpenAI LangChain integration
- `langchain-anthropic` - Anthropic LangChain integration
- `langchain-community` - Community LangChain components
- `openai` - Native OpenAI SDK (Responses API, Files API)
- `anthropic` - Native Anthropic SDK (web search, tool use)
- `replicate` - DeepSeek-R1 via Replicate
- `python-dotenv` - Environment variable management

## Troubleshooting

### "API key not found" errors
- Ensure `.chat-env` file exists in the project root
- Verify all required API keys are present and valid

### Response truncated before any text was produced
- Happens with GPT-5.4 + Medium/High reasoning + Web Search when the token budget is exhausted before the model produces output
- Try lowering the Reasoning Effort, disabling Web Search, or narrowing the question

### File upload not working
- For OpenAI: check file size limits (max 512 MB)
- Verify file permissions are readable

### Web search not working
- Claude models: verify the Anthropic API key has tool use enabled
- OpenAI models: web search is via the native Responses API — no external key needed
- Check internet connectivity

### Model not responding
- Check API key validity
- Verify model identifier is correct in the model dictionary
- Check for rate limiting or quota issues with the provider
