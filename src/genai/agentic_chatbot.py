from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Annotated, TypedDict
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

load_dotenv()

st.set_page_config(
    page_title="Assistant IA",
    page_icon="AI",
    layout="centered",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAISS_INDEX_PATH = PROJECT_ROOT / "notebooks" / "faiss_index"


search_runner = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic."""
    try:
        return search_runner.run(query)
    except Exception as error:
        return f"Error web_search: {error}"


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA').
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY is missing."

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper(),
                "apikey": api_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        quote = data.get("Global Quote", {})
        if not quote:
            message = (
                data.get("Note")
                or data.get("Error Message")
                or "No quote data returned."
            )
            return f"Unable to fetch stock price for {symbol.upper()}: {message}"

        price = quote.get("05. price", "n/a")
        change = quote.get("09. change", "n/a")
        change_percent = quote.get("10. change percent", "n/a")

        return (
            f"Latest price for {symbol.upper()}: {price} USD "
            f"(change: {change}, change percent: {change_percent})."
        )
    except requests.Timeout:
        return "The stock request timed out."
    except requests.RequestException as error:
        return f"Network error while fetching stock price: {error}"
    except ValueError:
        return "Alpha Vantage returned an invalid JSON response."


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city using the Weatherstack API.
    The input must contain only the city name.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Error: WEATHERSTACK_API_KEY is missing."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={
                "access_key": api_key,
                "query": city,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "current" not in data:
            error = data.get("error", {}).get(
                "info",
                "Invalid Weatherstack response.",
            )
            return f"Unable to get weather for {city}: {error}"

        location = data.get("location", {})
        current = data["current"]

        city_name = location.get("name", city)
        country = location.get("country", "")
        temperature = current.get("temperature")
        humidity = current.get("humidity")
        descriptions = current.get("weather_descriptions", [])
        description = descriptions[0] if descriptions else "unavailable"

        return (
            f"Current weather in {city_name}, {country}: "
            f"{temperature} C, {description}, humidity {humidity}%."
        )
    except requests.Timeout:
        return "The weather request timed out."
    except requests.RequestException as error:
        return f"Weather network error: {error}"
    except ValueError:
        return "Weatherstack returned an invalid JSON response."


@tool
def purchase_stock(symbol: str, quantity: int) -> str:
    """Ask a human to approve a stock purchase before simulating it."""

    # Pause here so a person can review the order.
    approval = interrupt(
        {
            "action": "purchase_stock",
            "message": f"Approve buying {quantity} shares of {symbol}?",
            "symbol": symbol,
            "quantity": quantity,
        }
    )

    # This code runs again after the user approves or rejects the order.
    if isinstance(approval, dict):
        decision = str(approval.get("decision", "reject")).lower()
        final_symbol = str(approval.get("symbol", symbol)).strip().upper() or symbol
        final_quantity = approval.get("quantity", quantity)
        reason = str(approval.get("reason", "")).strip()
    else:
        decision = "approve" if approval is True else "reject"
        final_symbol = symbol.upper()
        final_quantity = quantity
        reason = ""

    try:
        final_quantity = int(final_quantity)
    except (TypeError, ValueError):
        final_quantity = quantity

    # No real broker call happens here.
    # This is only a safe simulation for the chatbot demo.
    if decision == "approve":
        return (
            f"Purchase approved. Simulated order for {final_quantity} shares "
            f"of {final_symbol}."
        )

    if decision == "edit":
        return (
            f"Purchase edited and approved. Simulated order for "
            f"{final_quantity} shares of {final_symbol}."
        )

    if reason:
        return f"Purchase rejected by the human reviewer: {reason}"

    return "Purchase rejected by the human reviewer."


TOOL_SYSTEM_PROMPT = """
You are an assistant with native tool-calling enabled.
When a tool is required, use the model's tool call mechanism only.
Never output XML tags such as <function>...</function> or free-form pseudo tool syntax.
Never invent tool results.
If the user asks for live data, calculations, weather, stock prices, web search, or questions about the YC PDF knowledge base, prefer the appropriate tool.
If the user wants to buy stocks, use `purchase_stock` only after the request is clear.
Never say the stock was bought until the tool finishes and the human approves it.
Keep the final answer concise and grounded in the tool output.
""".strip()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _tool_call_value(tool_call: Any, key: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)


def _latest_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            return messages[index:]
    return messages


def _extract_tool_usage(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    usage: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    for message in messages:
        if getattr(message, "type", None) == "ai":
            for tool_call in getattr(message, "tool_calls", []) or []:
                tool_id = _tool_call_value(tool_call, "id")
                item = {
                    "id": tool_id,
                    "name": _tool_call_value(tool_call, "name", "tool"),
                    "args": _tool_call_value(tool_call, "args", {}),
                    "output": "",
                }
                usage.append(item)
                if tool_id:
                    by_id[str(tool_id)] = item

        elif getattr(message, "type", None) == "tool":
            tool_id = getattr(message, "tool_call_id", None)
            output = _message_content_to_text(getattr(message, "content", ""))

            if tool_id and str(tool_id) in by_id:
                by_id[str(tool_id)]["output"] = output
            else:
                usage.append(
                    {
                        "id": tool_id,
                        "name": getattr(message, "name", "tool"),
                        "args": {},
                        "output": output,
                    }
                )

    return usage


def _format_tool_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _extract_latest_human_text(messages: list[BaseMessage]) -> str:
    # Scan the latest human message first.
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _message_content_to_text(getattr(message, "content", ""))
    return ""


def _extract_stock_symbol(text: str) -> str | None:
    # First, try to read the symbol from the failed tool generation payload.
    patterns = [
        r'get_stock_price\{\"symbol\":\s*"([^"]+)"\}',
        r'"symbol":\s*"([^"]+)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            symbol = match.group(1).strip().upper()
            if symbol:
                return symbol

    # Then, try a simple ticker-like token from the text.
    ticker_match = re.search(r"\b[A-Z]{1,6}\b", text)
    if ticker_match:
        symbol = ticker_match.group(0).strip().upper()
        if symbol not in {"I", "A", "AN", "THE", "USD"}:
            return symbol

    return None


def _build_manual_stock_response(
    messages: list[BaseMessage],
    error: Exception,
) -> AIMessage | None:
    # Use the failed tool call first, then fall back to the user's message.
    error_text = str(error)
    symbol = _extract_stock_symbol(error_text)
    if not symbol:
        symbol = _extract_stock_symbol(_extract_latest_human_text(messages))

    if not symbol:
        return None

    try:
        stock_output = get_stock_price.invoke({"symbol": symbol})
    except Exception:
        return None

    # Keep the tool used by the fallback visible in the UI.
    return AIMessage(
        content=_message_content_to_text(stock_output),
        additional_kwargs={
            "manual_tool_usage": [
                {
                    "name": "get_stock_price",
                    "args": {"symbol": symbol},
                    "output": _message_content_to_text(stock_output),
                }
            ]
        },
    )


def _extract_interrupt_payloads(result: Any) -> list[dict[str, Any]]:
    # LangGraph can return interrupts as a dict field or as a GraphOutput object.
    raw_interrupts = []

    if isinstance(result, dict):
        raw_interrupts = result.get("__interrupt__", []) or []
    else:
        raw_interrupts = getattr(result, "interrupts", []) or []

    payloads: list[dict[str, Any]] = []

    for item in raw_interrupts:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            payloads.append(value)
        else:
            payloads.append({"message": str(value)})

    return payloads


def _get_result_messages(result: Any) -> list[BaseMessage]:
    if isinstance(result, dict):
        return result.get("messages", []) or []

    value = getattr(result, "value", None)
    if isinstance(value, dict):
        return value.get("messages", []) or []

    return []


def _get_manual_tool_usage(message: BaseMessage | None) -> list[dict[str, Any]]:
    if message is None:
        return []

    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    manual_tool_usage = additional_kwargs.get("manual_tool_usage", [])

    if isinstance(manual_tool_usage, list):
        return [
            item
            for item in manual_tool_usage
            if isinstance(item, dict)
        ]

    return []


def _run_turn(
    conversation: dict[str, Any],
    user_input: str | None = None,
    resume_value: Any | None = None,
) -> dict[str, Any]:
    # Use the same thread_id so the graph can resume the same conversation.
    config = {
        "configurable": {
            "thread_id": conversation["thread_id"],
        }
    }

    if resume_value is None:
        if user_input is None:
            raise ValueError("user_input is required when resume_value is missing.")

        result = workflow.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input),
                ]
            },
            config=config,
        )
    else:
        result = workflow.invoke(
            Command(resume=resume_value),
            config=config,
        )

    interrupts = _extract_interrupt_payloads(result)
    if interrupts:
        conversation["pending_hitl"] = interrupts
        return {
            "status": "interrupted",
            "interrupts": interrupts,
        }

    conversation["pending_hitl"] = []
    messages = _get_result_messages(result)
    latest_ai_message = next(
        (
            message
            for message in reversed(messages)
            if getattr(message, "type", None) == "ai"
        ),
        None,
    )

    assistant_response = _message_content_to_text(
        getattr(latest_ai_message, "content", "")
    )
    tool_usage = _get_manual_tool_usage(latest_ai_message)
    if not tool_usage:
        tool_usage = _extract_tool_usage(_latest_turn_messages(messages))

    conversation["messages"].append(
        {
            "role": "assistant",
            "content": assistant_response,
            "tools": tool_usage,
        }
    )

    return {
        "status": "completed",
        "assistant_response": assistant_response,
        "tool_usage": tool_usage,
    }


def _render_hitl_panel(conversation: dict[str, Any]) -> dict[str, Any] | None:
    pending = conversation.get("pending_hitl") or []
    if not pending:
        return None

    # We only expect one purchase review at a time in this app.
    review = pending[0] if isinstance(pending, list) else pending
    if not isinstance(review, dict):
        review = {"message": str(review)}

    st.warning("A stock purchase needs your approval.")
    st.write(review.get("message", "Review the pending action below."))

    default_symbol = str(review.get("symbol", "")).strip().upper()
    default_quantity = review.get("quantity", 1)

    try:
        default_quantity = int(default_quantity)
    except (TypeError, ValueError):
        default_quantity = 1

    # These keys keep the widgets stable across reruns.
    symbol = st.text_input(
        "Symbol",
        value=default_symbol,
        key=f"hitl_symbol_{conversation['id']}",
    )
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=default_quantity,
        step=1,
        key=f"hitl_quantity_{conversation['id']}",
    )
    reason = st.text_input(
        "Reject reason (optional)",
        key=f"hitl_reason_{conversation['id']}",
    )

    approve_col, reject_col = st.columns(2)

    with approve_col:
        approve_clicked = st.button(
            "Approve purchase",
            key=f"hitl_approve_{conversation['id']}",
            use_container_width=True,
        )

    with reject_col:
        reject_clicked = st.button(
            "Reject purchase",
            key=f"hitl_reject_{conversation['id']}",
            use_container_width=True,
        )

    if approve_clicked:
        # The human can edit the order before approving it.
        return {
            "decision": "approve",
            "symbol": symbol.strip().upper() or default_symbol,
            "quantity": int(quantity),
        }

    if reject_clicked:
        return {
            "decision": "reject",
            "reason": reason.strip(),
        }

    return None


@st.cache_resource
def build_rag_retriever():
    embeddings_model = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        provider="hf-inference",
    )

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run the notebook indexing step first."
        )

    vector_store = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings_model,
        allow_dangerous_deserialization=True,
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


rag_retriever = build_rag_retriever()


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the YC PDF document."""
    documents = rag_retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF"

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


tools = [web_search, calculator, get_stock_price, get_weather, rag_tool, purchase_stock]


def _is_tool_use_failed(error: Exception) -> bool:
    error_text = str(error).lower()

    if "tool_use_failed" in error_text or "failed_generation" in error_text:
        return True

    response = getattr(error, "response", None)
    if response is not None:
        response_text = (
            getattr(response, "text", None)
            or getattr(response, "content", None)
            or ""
        )
        response_text = str(response_text).lower()

        if "tool_use_failed" in response_text or "failed_generation" in response_text:
            return True

    return False


def _render_tool_usage(tool_usage: list[dict[str, Any]]) -> None:
    if not tool_usage:
        return

    tool_names = ", ".join(f"`{item['name']}`" for item in tool_usage)
    st.caption(f"Outils utilises: {tool_names}")

    with st.expander("Details des outils", expanded=False):
        for item in tool_usage:
            st.markdown(f"**{item['name']}**")

            args = item.get("args")
            if args:
                st.code(_format_tool_value(args), language="json")

            output = item.get("output")
            if output:
                st.code(_format_tool_value(output), language="text")


def _render_messages(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_tool_usage(message.get("tools", []))


@st.cache_resource
def create_workflow():
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    fallback_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
    )
    llm_with_tools = model.bind_tools(tools)
    llm_with_tools_fallback = fallback_model.bind_tools(tools)

    def chat_node(state: ChatState):
        messages = [
            SystemMessage(content=TOOL_SYSTEM_PROMPT),
            *state["messages"],
        ]

        attempts = [
            llm_with_tools,
            llm_with_tools_fallback,
        ]

        last_error: Exception | None = None

        for attempt_index, llm in enumerate(attempts, start=1):
            try:
                response = llm.invoke(messages)
                return {"messages": [response]}
            except Exception as error:
                last_error = error
                if _is_tool_use_failed(error) and attempt_index < len(attempts):
                    continue
                manual_response = _build_manual_stock_response(messages, error)
                if manual_response is not None:
                    return {"messages": [manual_response]}
                raise

        if last_error is not None:
            manual_response = _build_manual_stock_response(messages, last_error)
            if manual_response is not None:
                return {"messages": [manual_response]}
            raise last_error

        raise RuntimeError("Tool calling failed without a specific error.")

    checkpoint = MemorySaver()
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpoint)


workflow = create_workflow()


if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "conversation_counter" not in st.session_state:
    st.session_state.conversation_counter = 0

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = ""


def _create_conversation(select: bool = True) -> str:
    st.session_state.conversation_counter += 1

    conversation_id = str(uuid4())
    st.session_state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": f"Conversation {st.session_state.conversation_counter}",
        "thread_id": str(uuid4()),
        "messages": [],
        # Keep one pending approval per conversation.
        "pending_hitl": [],
    }

    if select:
        st.session_state.active_conversation_id = conversation_id

    return conversation_id


if not st.session_state.conversations:
    _create_conversation(select=True)

if st.session_state.active_conversation_id not in st.session_state.conversations:
    st.session_state.active_conversation_id = next(iter(st.session_state.conversations))


def _get_active_conversation() -> dict[str, Any]:
    return st.session_state.conversations[st.session_state.active_conversation_id]


def _maybe_refresh_conversation_title(conversation: dict[str, Any], user_input: str) -> None:
    if not conversation["title"].startswith("Conversation "):
        return

    snippet = " ".join(user_input.strip().split())
    if not snippet:
        return

    max_length = 42
    conversation["title"] = (
        snippet[:max_length] + "..."
        if len(snippet) > max_length
        else snippet
    )


with st.sidebar:
    st.header("Conversations")

    if st.button("Nouvelle conversation", use_container_width=True):
        _create_conversation(select=True)

    conversation_ids = list(reversed(list(st.session_state.conversations.keys())))
    active_id = st.session_state.active_conversation_id

    selected_conversation_id = st.radio(
        "Liste des conversations",
        options=conversation_ids,
        index=conversation_ids.index(active_id) if active_id in conversation_ids else 0,
        format_func=lambda conversation_id: st.session_state.conversations[conversation_id]["title"],
        label_visibility="collapsed",
    )

    st.session_state.active_conversation_id = selected_conversation_id


st.title("Assistant IA")

active_conversation = _get_active_conversation()

_render_messages(active_conversation["messages"])

resume_value = _render_hitl_panel(active_conversation)
if resume_value is not None:
    try:
        # Resume the same graph thread after the human decision.
        _run_turn(active_conversation, resume_value=resume_value)
        st.rerun()
    except Exception as error:
        st.error(f"Une erreur est survenue : {error}")


user_input = st.chat_input(
    "Ecrivez votre message...",
    disabled=bool(active_conversation.get("pending_hitl")),
)


if user_input:
    active_conversation = _get_active_conversation()

    # Save the user message before the graph runs.
    active_conversation["messages"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    _maybe_refresh_conversation_title(active_conversation, user_input)

    try:
        # Run the agent once for the new user message.
        _run_turn(active_conversation, user_input=user_input)
        st.rerun()
    except Exception as error:
        st.error(f"Une erreur est survenue : {error}")
