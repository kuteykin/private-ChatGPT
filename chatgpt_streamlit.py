# app.py
from dotenv import load_dotenv
import base64
import tempfile
import time
import os
from typing import Optional

from langchain_community.callbacks import get_openai_callback
from langchain.globals import set_verbose
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import replicate
import streamlit as st
from anthropic import Anthropic
from openai import OpenAI

# Model name mappings
ANTHROPIC_MODELS = {
    "Claude-Opus-5.5": "claude-opus-5-5",
    "Claude-Sonnet-5": "claude-sonnet-5",
}

ANTHROPIC_MODEL_ALIASES = {
    "Claude Opus-5.5": "claude-opus-5-5",
    "Claude Sonnet-5": "claude-sonnet-5",
}


def get_anthropic_api_model(selected_model: str) -> Optional[str]:
    if selected_model in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[selected_model]
    if selected_model in ANTHROPIC_MODEL_ALIASES:
        return ANTHROPIC_MODEL_ALIASES[selected_model]
    return None

OPENAI_MODELS = {
    "OpenAI GPT-6 Sol": "gpt-6-sol",
    "OpenAI GPT-6 Luna": "gpt-6-luna",
    "OpenAI-GPT-5.6 Terra": "gpt-5.6-terra",
    "EXPENSIVE OpenAI GPT-6 Astra": "gpt-6-astra",
}

DEEPSEEK_MODEL_LABEL = "DeepSeek-V3.1"
DEEPSEEK_REPLICATE_MODEL = "deepseek-ai/deepseek-v3.1"

# File type mappings
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
TEXT_EXTENSIONS = {"txt", "md", "csv"}

REASONING_EFFORT_OPTIONS = ("none", "low", "medium", "high")


def normalize_reasoning_effort(user_effort: str) -> str:
    """Accept current and legacy sidebar values."""
    legacy = {"None": "none", "Low": "low", "Medium": "medium", "High": "high"}
    if user_effort in legacy:
        return legacy[user_effort]
    if user_effort in REASONING_EFFORT_OPTIONS:
        return user_effort
    return "low"


def get_openai_reasoning_effort(user_effort: str) -> str:
    """Map sidebar selection to OpenAI Responses API reasoning.effort."""
    if user_effort in REASONING_EFFORT_OPTIONS:
        return user_effort
    return "medium"


def get_anthropic_reasoning_effort(user_effort: str) -> str:
    """Map sidebar selection to Anthropic output_config.effort (no API 'none')."""
    if user_effort == "none":
        return "low"
    if user_effort in ("low", "medium", "high"):
        return user_effort
    return "medium"


def apply_anthropic_output_effort(request_params: dict, effort: str) -> None:
    """Attach Anthropic effort via output_config on the HTTP request."""
    extra_body = dict(request_params.get("extra_body") or {})
    extra_body["output_config"] = {"effort": effort}
    request_params["extra_body"] = extra_body


def create_anthropic_message(client, request_params: dict):
    """Create a message, falling back if output_config.effort is rejected."""
    try:
        return client.messages.create(**request_params)
    except Exception as exc:
        if "output_config" not in str(exc) and "effort" not in str(exc):
            raise
        fallback = request_params.copy()
        extra_body = dict(fallback.pop("extra_body", None) or {})
        extra_body.pop("output_config", None)
        if extra_body:
            fallback["extra_body"] = extra_body
        return client.messages.create(**fallback)


def is_deepseek_model(selected_model: str = "") -> bool:
    return selected_model == DEEPSEEK_MODEL_LABEL


def format_deepseek_pdf_attachments(pdf_contents, pdf_filenames) -> str:
    """Embed PDF bytes for DeepSeek native document parsing in the prompt."""
    blocks = []
    for pdf_b64, filename in zip(pdf_contents, pdf_filenames):
        blocks.append(f'<pdf filename="{filename}">\n' f"data:application/pdf;base64,{pdf_b64}\n" f"</pdf>")
    return "\n\n".join(blocks)


def messages_to_deepseek_prompt(messages, pdf_contents=None, pdf_filenames=None) -> str:
    """Flatten LangChain messages into a single prompt for Replicate."""
    parts = []
    last_index = len(messages) - 1
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            parts.append(f"System:\n{msg.content}")
        elif isinstance(msg, HumanMessage):
            content = msg.content
            if pdf_contents and pdf_filenames and i == last_index and isinstance(msg, HumanMessage):
                content = format_deepseek_pdf_attachments(pdf_contents, pdf_filenames) + "\n\n" + content
            parts.append(f"User:\n{content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant:\n{msg.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def collect_replicate_output(output) -> str:
    if isinstance(output, str):
        return output.strip()
    if hasattr(output, "__iter__") and not isinstance(output, (str, bytes, dict)):
        return "".join(str(chunk) for chunk in output).strip()
    return str(output).strip()


def get_deepseek_thinking(user_effort: str) -> Optional[str]:
    """Map sidebar selection to Replicate DeepSeek `thinking` (separate from OpenAI/Anthropic)."""
    if user_effort in ("none", "low"):
        return None
    return "medium"


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


def is_openai_model(model_str: str = "", selected_model: str = "") -> bool:
    """Check if model is OpenAI-based"""
    if selected_model:
        return selected_model in OPENAI_MODELS
    return False


def is_anthropic_model(model_str: str = "", selected_model: str = "") -> bool:
    """Check if model is Anthropic-based"""
    if selected_model:
        return get_anthropic_api_model(selected_model) is not None
    return False


def get_image_mime_type(filename: str) -> str:
    """Get MIME type for image file"""
    ext = filename.lower().split(".")[-1]
    return f"image/{ext}" if ext != "jpg" else "image/jpeg"


def get_anthropic_workspace_id() -> Optional[str]:
    """Workspace ID for multi-workspace API keys (anthropic-workspace-id header)."""
    workspace = (os.getenv("ANTHROPIC_WORKSPACE") or "").strip()
    return workspace or None


def create_anthropic_client() -> Anthropic:
    """Anthropic client with optional workspace header from ANTHROPIC_WORKSPACE."""
    kwargs = {"api_key": os.getenv("ANTHROPIC_API_KEY")}
    workspace_id = get_anthropic_workspace_id()
    if workspace_id:
        kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
    return Anthropic(**kwargs)


def upload_file_to_anthropic(file_content: bytes, file_name: str) -> str:
    """Upload file to Anthropic's Files API and return file_id"""
    try:
        client = create_anthropic_client()

        # Determine file purpose based on type
        file_type = get_file_type(file_name)

        response = client.files.create(
            file=(file_name, file_content),
            # purpose is not needed for the current Files API
        )

        return response.id
    except Exception as e:
        st.error(f"Error uploading file to Anthropic: {str(e)}")
        return ""


def upload_file_to_openai(file_content: bytes, file_name: str) -> str:
    """Upload file to OpenAI's Files API and return file_id

    Note: After upload, the file needs to be processed before it can be used.
    This function waits for the file to be processed.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Create a temporary file to upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
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
                    return ""
                time.sleep(1)
                wait_time += 1

            # If we get here, file processing timed out
            st.warning(f"File processing is taking longer than expected. File ID: {file_id}")
            return file_id  # Return anyway, might still work

        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)

    except Exception as e:
        st.error(f"Error uploading file to OpenAI: {str(e)}")
        return ""


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant. Respond your answer in markdown format.")
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
            "OpenAI GPT-6 Sol",
            "OpenAI GPT-6 Luna",
            "OpenAI-GPT-5.6 Terra",
            "EXPENSIVE OpenAI GPT-6 Astra",
            "Claude-Opus-5.5",
            "Claude-Sonnet-5",
            DEEPSEEK_MODEL_LABEL,
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

    # Reasoning effort selection
    reasoning_effort = st.sidebar.radio(
        "Reasoning Effort:",
        REASONING_EFFORT_OPTIONS,
        index=1,
        format_func=str.capitalize,
        help=(
            "OpenAI: sent as reasoning.effort (none, low, medium, high). "
            "Anthropic: sent as output_config.effort; 'none' maps to low. "
            "DeepSeek on Replicate: medium/high enable thinking mode."
        ),
    )
    st.session_state.reasoning_effort = normalize_reasoning_effort(reasoning_effort)

    # Show info about web search availability
    if enable_web_search:
        if ai_model.startswith("Claude"):
            st.sidebar.info("✅ Native web search enabled for Claude")
        elif ai_model in OPENAI_MODELS:
            st.sidebar.info("✅ Web search enabled (OpenAI Responses API)")
        elif is_deepseek_model(ai_model):
            st.sidebar.warning("Web search is not available for DeepSeek on Replicate.")

    if is_deepseek_model(ai_model):
        st.sidebar.caption(
            "DeepSeek-V3.1 accepts PDF attachments natively (raw PDF in the prompt). "
            "Text/Markdown/CSV are inlined. Images display in the UI only."
        )

    if ai_model in OPENAI_MODELS:
        # OpenAI uses native Responses API only (see get_answer_openai_native)
        return None

    if ai_model.startswith("Claude") or is_deepseek_model(ai_model):
        # Native SDK paths only (see get_answer_*_native)
        return None


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
                for image_content, image_mime_type in zip(image_contents, image_mime_types):
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
    client = create_anthropic_client()
    anthropic_messages, system_content = convert_messages_to_anthropic(
        messages, pdf_contents, pdf_filenames, image_contents, image_mime_types
    )

    # Get reasoning effort from session state
    reasoning_effort = normalize_reasoning_effort(st.session_state.get("reasoning_effort", "low"))
    anthropic_effort = get_anthropic_reasoning_effort(reasoning_effort)

    # Prepare request parameters
    request_params = {
        "model": model_name,
        "max_tokens": 4096,
        "messages": anthropic_messages,
    }

    if system_content:
        request_params["system"] = system_content

    if enable_web_search:
        request_params["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]

    apply_anthropic_output_effort(request_params, anthropic_effort)

    # Handle tool use loop - continue until we get final text response
    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = create_anthropic_message(client, request_params)

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

        # Handle case where response might have text directly
        if not text_parts and hasattr(response, "text"):
            text_parts.append(response.text)

        # If we have text content, return it immediately (this is the final answer)
        if text_parts:
            # Join text parts and ensure proper markdown formatting
            result = "\n".join(text_parts).strip()
            return result if result else ""

        # If response stopped due to tool use, continue the conversation
        if response.stop_reason == "tool_use":
            # Add assistant message with tool use to continue conversation
            anthropic_messages.append({"role": "assistant", "content": response.content})

            # Update request_params to use updated messages for next iteration
            request_params["messages"] = anthropic_messages
            continue

        # If we get here, no text content and not tool use - break
        break

    # Return empty string if no response was generated
    return ""


def convert_messages_to_openai(messages, file_ids=None, image_contents=None, image_mime_types=None):
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
                for image_content, image_mime_type in zip(image_contents, image_mime_types):
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime_type};base64,{image_content}"},
                        }
                    )

            # Attach files if provided and this is the last message
            if file_ids and is_last_message:
                for file_id in file_ids:
                    message_content.append({"type": "file", "file": {"file_id": file_id}})

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


def convert_chat_content_to_responses_format(content):
    """Convert Chat Completions content format to Responses API format

    Chat Completions uses: type: "text", "image_url", "file"
    Responses API uses: type: "input_text", "input_image", "input_file"
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        converted = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")

                # Convert text blocks
                if item_type == "text":
                    converted.append({"type": "input_text", "text": item.get("text", "")})

                # Convert image_url blocks to input_image
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                    # Responses API expects input_image with image_url field
                    converted.append({"type": "input_image", "image_url": url})

                # Convert file blocks to input_file
                elif item_type == "file":
                    file_info = item.get("file", {})
                    file_id = file_info.get("file_id", "") if isinstance(file_info, dict) else str(file_info)
                    converted.append({"type": "input_file", "file_id": file_id})

                # Keep other types as-is (shouldn't happen, but just in case)
                else:
                    converted.append(item)
            else:
                converted.append(item)

        return converted if converted else content

    return content


def extract_openai_responses_text(response) -> list[str]:
    """Extract assistant text from an OpenAI Responses API response."""
    text_parts: list[str] = []
    if not hasattr(response, "output") or not response.output:
        return text_parts

    if not isinstance(response.output, list):
        return text_parts

    for output_item in response.output:
        if hasattr(output_item, "type"):
            if output_item.type == "message" and hasattr(output_item, "content"):
                content = output_item.content
                if isinstance(content, list):
                    for content_item in content:
                        if hasattr(content_item, "type") and content_item.type == "output_text":
                            if hasattr(content_item, "text"):
                                text_parts.append(content_item.text)
                        elif isinstance(content_item, dict) and (
                            content_item.get("type") == "output_text" or "text" in content_item
                        ):
                            text_parts.append(content_item.get("text", ""))
                        elif isinstance(content_item, str):
                            text_parts.append(content_item)
                elif hasattr(content, "text"):
                    text_parts.append(content.text)
                elif isinstance(content, str):
                    text_parts.append(content)
            elif hasattr(output_item, "text"):
                text_parts.append(output_item.text)
        elif isinstance(output_item, dict):
            if output_item.get("type") == "message":
                content = output_item.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                elif isinstance(content, str):
                    text_parts.append(content)
            elif "text" in output_item:
                text_parts.append(output_item["text"])

    return text_parts


def build_openai_responses_input(openai_messages):
    """Build Responses API input from Chat Completions-style messages."""
    responses_input = []
    for msg in openai_messages:
        role = msg.get("role")
        content = convert_chat_content_to_responses_format(msg.get("content", ""))
        if role == "system":
            continue
        if role in ("user", "assistant"):
            responses_input.append({"role": role, "content": content})
    return responses_input


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
    openai_messages = convert_messages_to_openai(messages, file_ids, image_contents, image_mime_types)

    if model_name not in OPENAI_MODELS.values():
        return "Unsupported OpenAI model"

    reasoning_effort = normalize_reasoning_effort(st.session_state.get("reasoning_effort", "low"))
    effort_value = get_openai_reasoning_effort(reasoning_effort)

    responses_input = build_openai_responses_input(openai_messages)
    if not responses_input:
        return "Error: No messages found"

    try:
        responses_params = {
            "model": model_name,
            "input": responses_input,
            "reasoning": {"effort": effort_value},
            "max_output_tokens": 16384,
        }
        if enable_web_search:
            responses_params["tools"] = [{"type": "web_search_preview"}]

        response = client.responses.create(**responses_params)

        text_parts = extract_openai_responses_text(response)
        if text_parts:
            result = "\n".join(text_parts).strip()
            return result if result else ""

        incomplete = getattr(response, "incomplete_details", None)
        if incomplete is not None:
            reason = getattr(incomplete, "reason", None) or (
                incomplete.get("reason") if isinstance(incomplete, dict) else "unknown"
            )
            if reason == "max_output_tokens":
                return (
                    "⚠️ Response truncated before any text was produced — reasoning "
                    "and web search consumed the entire `max_output_tokens` budget. "
                    "Try lowering the Reasoning Effort, disabling Web Search, or "
                    "narrowing the question."
                )
            return f"⚠️ Response incomplete: {reason}"

        return "⚠️ Model returned no text output."
    except Exception as e:
        return f"Error calling Responses API: {str(e)}"


def get_answer_deepseek_native(messages, pdf_contents=None, pdf_filenames=None):
    """Call DeepSeek-V3.1 on Replicate with optional thinking mode."""
    reasoning_effort = normalize_reasoning_effort(st.session_state.get("reasoning_effort", "low"))
    replicate_input = {
        "prompt": messages_to_deepseek_prompt(messages, pdf_contents, pdf_filenames),
        "max_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.9,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    thinking = get_deepseek_thinking(reasoning_effort)
    if thinking:
        replicate_input["thinking"] = thinking

    try:
        output = replicate.run(DEEPSEEK_REPLICATE_MODEL, input=replicate_input)
        result = collect_replicate_output(output)
        return result if result else "⚠️ DeepSeek returned no text output."
    except Exception as e:
        return f"Error calling DeepSeek on Replicate: {str(e)}"


def get_answer(llm, messages):
    """Get answer from LLM, handling web search for supported models, multiple PDF files, and images"""
    enable_web_search = st.session_state.get("enable_web_search", False)
    selected_model = st.session_state.get("selected_model", None)

    # Get file attachments from session state
    file_ids, pdf_contents, pdf_filenames, image_contents, image_mime_types = get_file_attachments()

    # Show status message if files are attached
    if file_ids or pdf_contents or image_contents:
        file_count = len(file_ids or []) + len(pdf_contents or []) + len(image_contents or [])
        st.info(f"📎 Sending {file_count} attached file(s) to {selected_model} for analysis...")

    if selected_model in OPENAI_MODELS:
        return get_answer_openai_native(
            messages,
            OPENAI_MODELS[selected_model],
            enable_web_search,
            file_ids,
            image_contents,
            image_mime_types,
        )

    if is_deepseek_model(selected_model or ""):
        return get_answer_deepseek_native(messages, pdf_contents, pdf_filenames)

    if selected_model and selected_model.startswith("Claude"):
        model_name = get_anthropic_api_model(selected_model or "")
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
        return (
            f"Error: Unknown Claude model '{selected_model}'. "
            "Pick Claude Opus 5.5 or Claude Sonnet 5 in the sidebar."
        )

    if llm is None:
        return "Error: No LLM selected."

    answer = llm.invoke(messages)

    if hasattr(answer, "content"):
        result = answer.content
        return result.strip() if isinstance(result, str) else str(result).strip()
    if isinstance(answer, str):
        return answer.strip()
    return str(answer).strip()


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
    if is_openai_model(model_str, selected_model) or is_anthropic_model(model_str, selected_model):
        st.session_state.last_image_contents.append(base64_image)
        st.session_state.last_image_mime_types.append(mime_type)
    elif is_deepseek_model(selected_model or ""):
        st.warning(f"{uploaded_file.name} will display in chat but DeepSeek-V3.1 cannot analyze images.")

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
                st.error(f"Failed to upload {uploaded_file.name} to OpenAI. Please try again.")
                user_input_prefix = f"📎 {uploaded_file.name} (upload failed)"
    elif is_anthropic_model(model_str, selected_model) or is_deepseek_model(selected_model or ""):
        with st.spinner(f"Preparing {uploaded_file.name}..."):
            try:
                pdf_base64 = encode_image_to_base64(file_content)
                st.session_state.last_pdf_contents.append(pdf_base64)
                st.session_state.last_pdf_filenames.append(uploaded_file.name)
            except Exception as e:
                st.error(f"Error preparing PDF: {str(e)}")
                user_input_prefix = f"📎 {uploaded_file.name} (preparation failed)"
    else:
        st.warning("PDF support not available for this model. Please use OpenAI or Anthropic models.")
        user_input_prefix = f"📎 {uploaded_file.name} (PDF support not available)"

    return user_input_prefix


def process_uploaded_files(uploaded_files, user_input, llm):
    """Process multiple uploaded files and return modified user input"""
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

    # If no files uploaded, only clear if we explicitly had files before
    # This allows files to persist across messages
    if not uploaded_files:
        return user_input

    model_str = str(type(llm))
    selected_model = st.session_state.get("selected_model")

    # Create a set of current file identifiers (name + size)
    current_file_ids = {(f.name, f.size) for f in uploaded_files}

    # Check if these are the same files as before
    if current_file_ids == st.session_state.processed_files:
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
            prefix = process_image_file(uploaded_file, file_content, model_str, selected_model)
            prefixes.append(prefix)
        elif file_type == "pdf":
            prefix = process_pdf_file(uploaded_file, file_content, model_str, selected_model)
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

    # Display currently attached files in sidebar
    if st.session_state.get("processed_files"):
        st.sidebar.markdown("### 📎 Currently Attached Files")
        file_names = [name for name, _ in st.session_state.processed_files]
        for name in file_names:
            st.sidebar.text(f"  • {name}")
        if st.sidebar.button("Clear All Attachments"):
            st.session_state.last_file_ids = []
            st.session_state.last_pdf_contents = []
            st.session_state.last_pdf_filenames = []
            st.session_state.last_image_contents = []
            st.session_state.last_image_mime_types = []
            st.session_state.last_images = []
            st.session_state.processed_files = set()
            st.rerun()

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
