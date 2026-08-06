import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from uuid import uuid4

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

st.set_page_config(
    page_title="Assistant IA",
    page_icon="🤖",
    layout="centered",
)


# ---------------------------------------------------------
# 1. Définition de l'état utilisé par LangGraph
# ---------------------------------------------------------
class ChatState(TypedDict):
    # add_messages permet d'ajouter les nouveaux messages
    # sans supprimer les anciens messages de la conversation.
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------
# 2. Création du modèle et du workflow LangGraph
# ---------------------------------------------------------
@st.cache_resource
def create_workflow():
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    def chat_node(state: ChatState):
        # Même si invoke() est utilisé dans le nœud,
        # LangGraph peut intercepter et streamer les tokens
        # avec workflow.stream(stream_mode="messages").
        response = model.invoke(state["messages"])

        return {
            "messages": [response]
        }

    # MemorySaver conserve le contexte selon le thread_id.
    checkpoint = MemorySaver()

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpoint)


workflow = create_workflow()


# ---------------------------------------------------------
# 3. Initialisation de la session Streamlit
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    # Chaque conversation possède son propre identifiant.
    st.session_state.thread_id = str(uuid4())

if "messages" not in st.session_state:
    # Cet historique sert uniquement à l'affichage dans Streamlit.
    st.session_state.messages = []


st.title("Assistant IA")
st.caption("Propulsé par LangGraph et Groq")


# ---------------------------------------------------------
# 4. Affichage de l'historique
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# 5. Récupération du message de l'utilisateur
# ---------------------------------------------------------
user_input = st.chat_input("Écrivez votre message...")


if user_input:
    # Ajouter le message à l'historique visuel Streamlit.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # LangGraph utilise ce thread_id pour retrouver
    # les anciens messages stockés par MemorySaver.
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    try:
        with st.chat_message("assistant"):

            # Ce conteneur sera mis à jour à chaque token reçu.
            response_placeholder = st.empty()

            # Variable qui contiendra progressivement
            # la réponse complète de l'assistant.
            assistant_response = ""

            # -------------------------------------------------
            # PARTIE IMPORTANTE : STREAMING DE LA RÉPONSE
            # -------------------------------------------------
            #
            # stream_mode="messages" demande à LangGraph
            # de renvoyer les morceaux de messages générés
            # par le modèle au fur et à mesure.
            #
            # Chaque élément retourné contient :
            # - message_chunk : un morceau de la réponse
            # - metadata : les informations sur le nœud exécuté
            #
            for message_chunk, metadata in workflow.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config,
                stream_mode="messages",
            ):
                # On s'assure que le contenu reçu est bien du texte.
                if isinstance(message_chunk.content, str):
                    token = message_chunk.content

                    # Ajouter le nouveau token à la réponse.
                    assistant_response += token

                    # Réafficher toute la réponse accumulée.
                    # Le symbole ▌ simule un curseur pendant l'écriture.
                    response_placeholder.markdown(
                        assistant_response + "▌"
                    )

            # Lorsque le streaming est terminé,
            # on retire le curseur ▌.
            response_placeholder.markdown(assistant_response)

        # Enregistrer la réponse complète dans l'historique visuel.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

    except Exception as error:
        st.error(f"Une erreur est survenue : {error}")


# ---------------------------------------------------------
# 6. Nouvelle conversation
# ---------------------------------------------------------
if st.button("Nouvelle conversation"):
    # Effacer l'historique affiché.
    st.session_state.messages = []

    # Générer un nouveau thread_id afin que LangGraph
    # ne récupère pas le contexte de l'ancienne conversation.
    st.session_state.thread_id = str(uuid4())

    # Relancer immédiatement l'interface.
    st.rerun()