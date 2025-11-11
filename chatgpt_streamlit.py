# app.py
from dotenv import load_dotenv
import base64
import tempfile
import time
import json
import os

from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Replicate
from langchain.globals import set_verbose
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from tavily import TavilyClient

# Model name mappings
ANTHROPIC_MODELS = {
    "Claude-4.1-Opus": "claude-opus-4-1-20250805",
    "Claude-4.5-Sonnet": "claude-sonnet-4-5",
}

OPENAI_MODELS = {
    "OpenAI-GPT-5": "gpt-5",
    "OpenAI-GPT-4.1": "gpt-4.1-2025-04-14",
    "OpenAI-Reasoning-o4-mini": "o4-mini",
}

# File type mappings
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
TEXT_EXTENSIONS = {"txt", "md", "csv"}


def init_page():
    st.set_page_config(page_title="Private ChatGPT")
    st.header("Private ChatGPT 🧐 by Dr.Konstantin Kuteykin-Teplyakov 👨‍🎓 ")
    st.sidebar.title("Options")


def encode_image_to_base64(image_bytes) -> str:
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")


def get_file_type(file_name: str) -> str:
    """Determine file type from extension"""
    ext = file_name.lower().split(".")[-1]
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext == "pdf":
        return "pdf"
    elif ext in TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def is_openai_model(model_str: str = None, selected_model: str = None) -> bool:
    """Check if model is OpenAI-based"""
    if selected_model:
        return selected_model.startswith("OpenAI")
    return "ChatOpenAI" in (model_str or "")


def is_anthropic_model(model_str: str = None, selected_model: str = None) -> bool:
    """Check if model is Anthropic-based"""
    if selected_model:
        return selected_model.startswith("Claude")
    return "ChatAnthropic" in (model_str or "")


def get_image_mime_type(filename: str) -> str:
    """Get MIME type for image file"""
    ext = filename.lower().split(".")[-1]
    return f"image/{ext}" if ext != "jpg" else "image/jpeg"


def web_search(query: str) -> str:
    """Perform web search using Tavily API

    Args:
        query: The search query string

    Returns:
        Formatted string with search results including titles, content, and URLs
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY not found in environment variables. Please add it to your .chat-env file."

        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query, max_results=5)

        if not response.get("results"):
            return f"No search results found for query: {query}"

        # Format results
        results = []
        for i, result in enumerate(response["results"], 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content available")
            url = result.get("url", "")
            results.append(f"{i}. **{title}**\n{content}\nSource: {url}")

        return "\n\n".join(results)
    except Exception as e:
        return f"Web search error: {str(e)}"


def upload_file_to_anthropic(file_content: bytes, file_name: str) -> str:
    """Upload file to Anthropic's Files API and return file_id"""
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Determine file purpose based on type
        file_type = get_file_type(file_name)

        response = client.files.create(
            file=(file_name, file_content),
            # purpose is not needed for the current Files API
        )

        return response.id
    except Exception as e:
        st.error(f"Error uploading file to Anthropic: {str(e)}")
        return None


def upload_file_to_openai(file_content: bytes, file_name: str) -> str:
    """Upload file to OpenAI's Files API and return file_id

    Note: After upload, the file needs to be processed before it can be used.
    This function waits for the file to be processed.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Create a temporary file to upload
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file_name)[1]
        ) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Upload to OpenAI
            with open(tmp_file_path, "rb") as f:
                file_obj = client.files.create(file=f, purpose="user_data")

            file_id = file_obj.id

            # Wait for file to be processed (required before use)
            max_wait = 60  # Maximum wait time in seconds
            wait_time = 0
            while wait_time < max_wait:
                file_status = client.files.retrieve(file_id)
                if file_status.status == "processed":
                    return file_id
                elif file_status.status == "error":
                    st.error(f"File processing failed: {file_status.error}")
                    return None
                time.sleep(1)
                wait_time += 1

            # If we get here, file processing timed out
            st.warning(
                f"File processing is taking longer than expected. File ID: {file_id}"
            )
            return file_id  # Return anyway, might still work

        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)

    except Exception as e:
        st.error(f"Error uploading file to OpenAI: {str(e)}")
        return None


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Respond your answer in markdown format."
            )
        ]
        # Clear file attachments when conversation is cleared
        st.session_state.last_file_ids = []
        st.session_state.last_pdf_contents = []
        st.session_state.last_pdf_filenames = []
        st.session_state.last_image_contents = []
        st.session_state.last_image_mime_types = []
        st.session_state.last_images = []
        st.session_state.processed_files = set()


def select_model():
    ai_model = st.sidebar.radio(
        "Choose LLM:",
        (
            "OpenAI-GPT-5",
            "OpenAI-GPT-4.1",
            "OpenAI-Reasoning-o4-mini",
            "Claude-4.1-Opus",
            "Claude-4.5-Sonnet",
            "DeepSeek-R1",
        ),
    )

    # Store selected model for file handling
    st.session_state.selected_model = ai_model

    # Web search toggle
    enable_web_search = st.sidebar.checkbox(
        "Enable Web Search",
        value=False,
        help="Allow the model to search the web for up-to-date information. By default, the model uses its internal knowledge.",
    )
    st.session_state.enable_web_search = enable_web_search

    # Show info about web search availability
    if enable_web_search:
        if ai_model.startswith("Claude"):
            st.sidebar.info("✅ Native web search enabled for Claude")
        elif ai_model.startswith("OpenAI"):
            st.sidebar.info("✅ Web search enabled via Tavily API")

    if ai_model == "OpenAI-Reasoning-o4-mini":
        return ChatOpenAI(
            model=OPENAI_MODELS["OpenAI-Reasoning-o4-mini"],
            use_responses_api=True,
            max_completion_tokens=8192,
            model_kwargs={
                "reasoning": {
                    "effort": "medium",  # 'low', 'medium', or 'high'
                    "summary": "auto",  # 'detailed', 'auto', or None
                },
            },
        )
    elif ai_model in OPENAI_MODELS:
        model_name = OPENAI_MODELS[ai_model]
        if ai_model == "OpenAI-GPT-5":
            return ChatOpenAI(model_name=model_name)
        elif ai_model == "OpenAI-GPT-4.1":
            return ChatOpenAI(temperature=0.0, model_name=model_name)
    elif ai_model.startswith("Claude"):
        model_name = ANTHROPIC_MODELS.get(ai_model)
        return ChatAnthropic(temperature=0.0, max_tokens=4096, model=model_name)
    elif ai_model == "DeepSeek-R1":
        return Replicate(
            streaming=True,
            model_kwargs={
                "temperature": 0.0,
                "max_new_tokens": 8192,
                "top_p": 0.9,
            },
            model="deepseek-ai/deepseek-r1",
        )


def convert_messages_to_anthropic(
    messages,
    pdf_contents=None,
    pdf_filenames=None,
    image_contents=None,
    image_mime_types=None,
):
    """Convert LangChain messages to Anthropic format with multiple file support"""
    anthropic_messages = []
    system_content = None

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_content = msg.content
        elif isinstance(msg, HumanMessage):
            content = msg.content
            message_content = []
            is_last_message = msg == messages[-1]

            # Attach PDFs if provided and this is the last message
            if pdf_contents and pdf_filenames and is_last_message:
                for pdf_content, pdf_filename in zip(pdf_contents, pdf_filenames):
                    message_content.append(
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_content,
                            },
                        }
                    )

            # Attach images if provided and this is the last message
            if image_contents and image_mime_types and is_last_message:
                for image_content, image_mime_type in zip(
                    image_contents, image_mime_types
                ):
                    message_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_mime_type,
                                "data": image_content,
                            },
                        }
                    )

            # Add text content
            if message_content:
                message_content.append({"type": "text", "text": content})
                anthropic_messages.append({"role": "user", "content": message_content})
            else:
                anthropic_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            anthropic_messages.append({"role": "assistant", "content": msg.content})

    return anthropic_messages, system_content


def get_answer_anthropic_native(
    messages,
    model_name,
    enable_web_search,
    pdf_contents=None,
    pdf_filenames=None,
    image_contents=None,
    image_mime_types=None,
):
    """Get answer from Anthropic using native SDK with web search, multiple PDFs and images support"""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    anthropic_messages, system_content = convert_messages_to_anthropic(
        messages, pdf_contents, pdf_filenames, image_contents, image_mime_types
    )

    # Prepare request parameters
    request_params = {
        "model": model_name,
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": anthropic_messages,
    }

    if system_content:
        request_params["system"] = system_content

    if enable_web_search:
        request_params["tools"] = [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
        ]

    # Handle tool use loop - continue until we get final text response
    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Make API call
        response = client.messages.create(**request_params)

        # First, extract only text content blocks (filter out all tool use and metadata)
        text_parts = []
        if response.content:
            for content_block in response.content:
                # Only extract text blocks, explicitly skip everything else
                if hasattr(content_block, "type"):
                    if content_block.type == "text" and hasattr(content_block, "text"):
                        text_parts.append(content_block.text)
                    # Skip all other types: tool_use, tool_result, etc.
                elif isinstance(content_block, dict):
                    if content_block.get("type") == "text":
                        text_parts.append(content_block.get("text", ""))
                    # Skip all other types

        # If we have text content, return it immediately (this is the final answer)
        if text_parts:
            return "\n".join(text_parts)

        # If response stopped due to tool use, continue the conversation
        if response.stop_reason == "tool_use":
            # Add assistant message with tool use to continue conversation
            anthropic_messages.append(
                {"role": "assistant", "content": response.content}
            )

            # Update request_params to use updated messages for next iteration
            request_params["messages"] = anthropic_messages
            continue

        # If we get here, no text content and not tool use - break
        break

    return ""


def convert_messages_to_openai(
    messages, file_ids=None, image_contents=None, image_mime_types=None
):
    """Convert LangChain messages to OpenAI format with multiple file support"""
    openai_messages = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            openai_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            content = msg.content
            message_content = []
            is_last_message = msg == messages[-1]

            # Attach images if provided and this is the last message
            if image_contents and image_mime_types and is_last_message:
                for image_content, image_mime_type in zip(
                    image_contents, image_mime_types
                ):
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_mime_type};base64,{image_content}"
                            },
                        }
                    )

            # Attach files if provided and this is the last message
            if file_ids and is_last_message:
                for file_id in file_ids:
                    message_content.append(
                        {"type": "file", "file": {"file_id": file_id}}
                    )

            # Add text content
            if message_content:
                if content:
                    message_content.insert(0, {"type": "text", "text": content})
                openai_messages.append({"role": "user", "content": message_content})
            else:
                openai_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            openai_messages.append({"role": "assistant", "content": msg.content})

    return openai_messages


def get_answer_openai_native(
    messages,
    model_name,
    enable_web_search,
    file_ids=None,
    image_contents=None,
    image_mime_types=None,
):
    """Get answer from OpenAI using native SDK with multiple PDF files, images support, and web search"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openai_messages = convert_messages_to_openai(
        messages, file_ids, image_contents, image_mime_types
    )

    # Prepare request parameters
    request_params = {
        "model": model_name,
        "messages": openai_messages,
    }

    if model_name != OPENAI_MODELS["OpenAI-GPT-5"]:
        request_params["temperature"] = 0.0

    if enable_web_search:
        request_params["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information, news, and up-to-date data. Use this when you need information beyond your knowledge cutoff date or for real-time information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up on the web",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    # Handle tool use loop - continue until we get final text response
    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Make API call
        response = client.chat.completions.create(**request_params)
        message = response.choices[0].message

        # If no tool calls, return the content
        if not message.tool_calls:
            return message.content

        # Process tool calls
        openai_messages.append(message)

        for tool_call in message.tool_calls:
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                search_results = web_search(args.get("query", ""))
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_results,
                    }
                )

        # Update request_params with new messages for next iteration
        request_params["messages"] = openai_messages

    # If we hit max iterations, return last message content
    return (
        message.content
        if message.content
        else "Unable to complete response after multiple tool calls."
    )


def get_answer(llm, messages):
    """Get answer from LLM, handling web search for supported models, multiple PDF files, and images"""
    enable_web_search = st.session_state.get("enable_web_search", False)
    selected_model = st.session_state.get("selected_model", None)

    # Get file attachments from session state
    file_ids, pdf_contents, pdf_filenames, image_contents, image_mime_types = (
        get_file_attachments()
    )

    # Use native SDK for web search or file handling
    if enable_web_search or file_ids or pdf_contents or image_contents:
        if selected_model and selected_model.startswith("Claude"):
            model_name = ANTHROPIC_MODELS.get(selected_model)
            if model_name:
                return get_answer_anthropic_native(
                    messages,
                    model_name,
                    enable_web_search,
                    pdf_contents,
                    pdf_filenames,
                    image_contents,
                    image_mime_types,
                )
        elif selected_model and selected_model.startswith("OpenAI"):
            model_name = OPENAI_MODELS.get(
                selected_model, OPENAI_MODELS["OpenAI-GPT-4.1"]
            )
            return get_answer_openai_native(
                messages,
                model_name,
                enable_web_search,
                file_ids,
                image_contents,
                image_mime_types,
            )

    # Default: use LangChain wrapper
    answer = llm.invoke(messages)
    return answer if isinstance(llm, Replicate) else answer.text()


def process_image_file(uploaded_file, file_content, model_str, selected_model):
    """Process uploaded image file and add to session state lists"""
    base64_image = encode_image_to_base64(file_content)
    mime_type = get_image_mime_type(uploaded_file.name)

    # Initialize lists if they don't exist
    if "last_images" not in st.session_state:
        st.session_state.last_images = []
    if "last_image_contents" not in st.session_state:
        st.session_state.last_image_contents = []
    if "last_image_mime_types" not in st.session_state:
        st.session_state.last_image_mime_types = []

    # Store image info for display
    st.session_state.last_images.append(
        {
            "name": uploaded_file.name,
            "data_uri": f"data:{mime_type};base64,{base64_image}",
        }
    )

    # Store for API use if model supports it
    if is_openai_model(model_str, selected_model) or is_anthropic_model(
        model_str, selected_model
    ):
        st.session_state.last_image_contents.append(base64_image)
        st.session_state.last_image_mime_types.append(mime_type)

    return f"📎 {uploaded_file.name} (image)"


def process_pdf_file(uploaded_file, file_content, model_str, selected_model):
    """Process uploaded PDF file and add to session state lists"""
    user_input_prefix = f"📎 PDF: {uploaded_file.name}"

    # Initialize lists if they don't exist
    if "last_file_ids" not in st.session_state:
        st.session_state.last_file_ids = []
    if "last_pdf_contents" not in st.session_state:
        st.session_state.last_pdf_contents = []
    if "last_pdf_filenames" not in st.session_state:
        st.session_state.last_pdf_filenames = []

    if is_openai_model(model_str, selected_model):
        with st.spinner(f"Uploading {uploaded_file.name} to OpenAI..."):
            file_id = upload_file_to_openai(file_content, uploaded_file.name)
            if file_id:
                st.session_state.last_file_ids.append(file_id)
            else:
                st.error(
                    f"Failed to upload {uploaded_file.name} to OpenAI. Please try again."
                )
                user_input_prefix = f"📎 {uploaded_file.name} (upload failed)"
    elif is_anthropic_model(model_str, selected_model):
        with st.spinner(f"Preparing {uploaded_file.name} for Claude..."):
            try:
                pdf_base64 = encode_image_to_base64(file_content)
                st.session_state.last_pdf_contents.append(pdf_base64)
                st.session_state.last_pdf_filenames.append(uploaded_file.name)
            except Exception as e:
                st.error(f"Error preparing PDF for Anthropic: {str(e)}")
                user_input_prefix = f"📎 {uploaded_file.name} (preparation failed)"
    else:
        st.warning(
            "PDF support not available for this model. Please use OpenAI or Anthropic models."
        )
        user_input_prefix = f"📎 {uploaded_file.name} (PDF support not available)"

    return user_input_prefix


def process_uploaded_files(uploaded_files, user_input, llm):
    """Process multiple uploaded files and return modified user input"""
    if not uploaded_files:
        # If no files but we had files before, clear them
        if st.session_state.get("processed_files"):
            st.session_state.last_file_ids = []
            st.session_state.last_pdf_contents = []
            st.session_state.last_pdf_filenames = []
            st.session_state.last_image_contents = []
            st.session_state.last_image_mime_types = []
            st.session_state.last_images = []
            st.session_state.processed_files = set()
        return user_input

    model_str = str(type(llm))
    selected_model = st.session_state.get("selected_model")

    # Initialize session state for file tracking
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "last_file_ids" not in st.session_state:
        st.session_state.last_file_ids = []
    if "last_pdf_contents" not in st.session_state:
        st.session_state.last_pdf_contents = []
    if "last_pdf_filenames" not in st.session_state:
        st.session_state.last_pdf_filenames = []
    if "last_image_contents" not in st.session_state:
        st.session_state.last_image_contents = []
    if "last_image_mime_types" not in st.session_state:
        st.session_state.last_image_mime_types = []
    if "last_images" not in st.session_state:
        st.session_state.last_images = []

    # Create a set of current file identifiers (name + size)
    current_file_ids = {(f.name, f.size) for f in uploaded_files}

    # Check if these are the same files as before
    if current_file_ids == st.session_state.processed_files:
        # Files haven't changed, don't reprocess - just return user input
        return user_input

    # Determine which files are new and which were removed
    new_file_ids = current_file_ids - st.session_state.processed_files
    removed_file_ids = st.session_state.processed_files - current_file_ids

    # If files were removed, we need to rebuild everything
    if removed_file_ids:
        # Clear all and reprocess remaining files
        st.session_state.last_file_ids = []
        st.session_state.last_pdf_contents = []
        st.session_state.last_pdf_filenames = []
        st.session_state.last_image_contents = []
        st.session_state.last_image_mime_types = []
        st.session_state.last_images = []
        st.session_state.processed_files = set()
        # Process all current files as new
        new_file_ids = current_file_ids

    # Process only new files
    prefixes = []
    text_content_parts = []

    for uploaded_file in uploaded_files:
        file_id = (uploaded_file.name, uploaded_file.size)

        # Skip files that have already been processed
        if file_id not in new_file_ids:
            continue

        file_content = uploaded_file.read()
        file_type = get_file_type(uploaded_file.name)

        if file_type == "image":
            prefix = process_image_file(
                uploaded_file, file_content, model_str, selected_model
            )
            prefixes.append(prefix)
        elif file_type == "pdf":
            prefix = process_pdf_file(
                uploaded_file, file_content, model_str, selected_model
            )
            prefixes.append(prefix)
        elif file_type == "text":
            # For text files, include content inline
            try:
                text = file_content.decode("utf-8")
                text_content_parts.append(f"📎 File: {uploaded_file.name}\n\n{text}")
            except Exception as e:
                st.error(f"Error reading text file {uploaded_file.name}: {str(e)}")
                prefixes.append(f"📎 {uploaded_file.name} (error reading)")
        else:
            prefixes.append(f"📎 {uploaded_file.name}")

    # Update processed files set
    st.session_state.processed_files = current_file_ids

    # Build final user input - only include prefixes for newly processed files
    result = ""
    if prefixes:
        result += ", ".join(prefixes) + "\n\n"
    if text_content_parts:
        result += "\n\n".join(text_content_parts) + "\n\n---\n\n"
    result += user_input

    return result


def get_file_attachments():
    """Get file attachments from session state for API calls"""
    file_ids = st.session_state.get("last_file_ids", [])
    pdf_contents = st.session_state.get("last_pdf_contents", [])
    pdf_filenames = st.session_state.get("last_pdf_filenames", [])
    image_contents = st.session_state.get("last_image_contents", [])
    image_mime_types = st.session_state.get("last_image_mime_types", [])

    # Return None for empty lists to maintain backward compatibility
    return (
        file_ids if file_ids else None,
        pdf_contents if pdf_contents else None,
        pdf_filenames if pdf_filenames else None,
        image_contents if image_contents else None,
        image_mime_types if image_mime_types else None,
    )


def display_chat_history():
    """Display chat history with messages and file attachments"""
    messages = st.session_state.get("messages", [])
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

                # Display all images if present for the last message
                if i == len(messages) - 1:  # Only for the most recent message
                    images = st.session_state.get("last_images", [])
                    if images:
                        for image_info in images:
                            st.image(image_info["data_uri"], caption=image_info["name"])


def main():
    load_dotenv(".chat-env")
    set_verbose(True)
    init_page()
    llm = select_model()
    init_messages()

    uploaded_files = st.file_uploader(
        "Upload PDF, images, or text files",
        type=["pdf", "png", "jpg", "jpeg", "gif", "webp", "txt", "md", "csv"],
        help="Upload documents or images to discuss with the AI",
        accept_multiple_files=True,
    )

    if user_input := st.chat_input("Input your question!"):
        # Process files if uploaded
        user_input = process_uploaded_files(uploaded_files, user_input, llm)
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.spinner("AI Chat Assistant is typing ..."):
            answer = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))

    display_chat_history()


if __name__ == "__main__":
    main()
