import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Assistant de recherche",
    page_icon="🤖",
    layout="centered",
)


# ---------------------------------------------------------
# Outils
# ---------------------------------------------------------

search_tool = DuckDuckGoSearchRun(
    name="internet_search",
    description=(
        "Recherche des informations récentes sur Internet. "
        "Utilise cet outil pour les actualités, les capitales, "
        "les événements récents et les informations générales."
    ),
)


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city using the Weatherstack API.
    The input must contain only the city name.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Erreur : la variable WEATHERSTACK_API_KEY est absente."

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
                "Réponse Weatherstack invalide.",
            )
            return f"Impossible d'obtenir la météo de {city} : {error}"

        location = data.get("location", {})
        current = data["current"]

        city_name = location.get("name", city)
        country = location.get("country", "")
        temperature = current.get("temperature")
        humidity = current.get("humidity")
        descriptions = current.get("weather_descriptions", [])
        description = descriptions[0] if descriptions else "indisponible"

        return (
            f"Météo actuelle à {city_name}, {country} : "
            f"{temperature} °C, {description}, "
            f"humidité de {humidity} %."
        )

    except requests.Timeout:
        return "La requête météo a expiré."

    except requests.RequestException as error:
        return f"Erreur réseau Weatherstack : {error}"

    except ValueError:
        return "Weatherstack a retourné une réponse JSON invalide."


# ---------------------------------------------------------
# Création de l’agent
# ---------------------------------------------------------

@st.cache_resource
def create_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "La variable GROQ_API_KEY est absente du fichier .env."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=groq_api_key,
    )

    client = Client()

    prompt = client.pull_prompt(
        "hwchase17/react",
        dangerously_pull_public_prompt=True,
    )

    tools = [search_tool, get_weather]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )


# ---------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------

st.title("🤖 Assistant de recherche")
st.caption("Recherche web et météo en temps réel")

with st.sidebar:
    st.header("Outils disponibles")

    st.write("🌐 Recherche avec DuckDuckGo")
    st.write("🌤️ Météo avec Weatherstack")

    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Posez-moi une question. Par exemple : "
                "« Quelle est la capitale de la RDC et quelle météo "
                "fait-il actuellement dans cette ville ? »"
            ),
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Écrivez votre question...")


if user_input:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                agent_executor = create_agent()

                response = agent_executor.invoke(
                    {
                        "input": user_input
                    }
                )

                answer = response.get(
                    "output",
                    "L’agent n’a retourné aucune réponse.",
                )

                st.markdown(answer)

            except Exception as error:
                answer = f"Une erreur est survenue : `{error}`"
                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )