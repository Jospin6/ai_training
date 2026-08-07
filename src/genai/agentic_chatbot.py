from __future__ import annotations

import json
import math
import os
from typing import Any, Annotated, TypedDict
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

st.set_page_config(
    page_title="Assistant IA",
    page_icon="AI",
    layout="centered",
)


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


tools = [web_search, calculator, get_stock_price, get_weather]

TOOL_SYSTEM_PROMPT = """
You are an assistant with native tool-calling enabled.
When a tool is required, use the model's tool call mechanism only.
Never output XML tags such as <function>...</function> or free-form pseudo tool syntax.
Never invent tool results.
If the user asks for live data, calculations, weather, stock prices, or web search, prefer the appropriate tool.
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
                raise

        if last_error is not None:
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


user_input = st.chat_input("Ecrivez votre message...")


if user_input:
    active_conversation = _get_active_conversation()

    active_conversation["messages"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    _maybe_refresh_conversation_title(active_conversation, user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": active_conversation["thread_id"],
        }
    }

    try:
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                result = workflow.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_input),
                        ]
                    },
                    config=config,
                )

            latest_ai_message = next(
                (
                    message
                    for message in reversed(result["messages"])
                    if getattr(message, "type", None) == "ai"
                ),
                None,
            )
            assistant_response = _message_content_to_text(
                getattr(latest_ai_message, "content", "")
            )

            turn_messages = _latest_turn_messages(result["messages"])
            tool_usage = _extract_tool_usage(turn_messages)

            st.markdown(assistant_response)
            _render_tool_usage(tool_usage)

        active_conversation["messages"].append(
            {
                "role": "assistant",
                "content": assistant_response,
                "tools": tool_usage,
            }
        )

    except Exception as error:
        st.error(f"Une erreur est survenue : {error}")
