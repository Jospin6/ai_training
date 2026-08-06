from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"


# ---------------------------------------------------------
# Chargement des fichiers
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    required_files = {
        "model": ARTIFACTS_DIR / "model.pkl",
        "book_names": ARTIFACTS_DIR / "book_names.pkl",
        "book_pivot": ARTIFACTS_DIR / "book_pivot.pkl",
        "final_rating": ARTIFACTS_DIR / "final_rating.pkl",
    }

    missing_files = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Fichiers introuvables :\n" + "\n".join(missing_files)
        )

    with required_files["model"].open("rb") as file:
        model = pickle.load(file)

    with required_files["book_names"].open("rb") as file:
        book_names = pickle.load(file)

    with required_files["book_pivot"].open("rb") as file:
        book_pivot = pickle.load(file)

    with required_files["final_rating"].open("rb") as file:
        final_rating = pickle.load(file)

    return model, book_names, book_pivot, final_rating


try:
    model, book_names, book_pivot, final_rating = load_artifacts()
except Exception as error:
    st.error(f"Impossible de charger les fichiers du modèle : {error}")
    st.stop()


# ---------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------
def get_column(column_name: str) -> str | None:
    return column_name if column_name in final_rating.columns else None


TITLE_COLUMN = get_column("title")
AUTHOR_COLUMN = get_column("author")
IMAGE_COLUMN = get_column("img_url")


def get_book_information(book_title: str) -> dict:
    information = {
        "title": book_title,
        "author": None,
        "image": None,
    }

    if TITLE_COLUMN is None:
        return information

    rows = final_rating[
        final_rating[TITLE_COLUMN].astype(str).str.strip()
        == str(book_title).strip()
    ]

    if rows.empty:
        return information

    first_row = rows.iloc[0]

    if AUTHOR_COLUMN is not None:
        author = first_row.get(AUTHOR_COLUMN)

        if author is not None and not np.isscalar(author):
            author = str(author)

        if author is not None and str(author).lower() != "nan":
            information["author"] = str(author)

    if IMAGE_COLUMN is not None:
        image = first_row.get(IMAGE_COLUMN)

        if image is not None and str(image).lower() != "nan":
            information["image"] = str(image)

    return information


def recommend_books(book_title: str, number_of_books: int = 5) -> list[dict]:
    pivot_titles = np.asarray(book_pivot.index).astype(str)
    matches = np.where(pivot_titles == str(book_title))[0]

    if len(matches) == 0:
        raise ValueError(
            f'Le livre "{book_title}" est absent de book_pivot.'
        )

    book_index = int(matches[0])
    book_vector = book_pivot.iloc[book_index].values.reshape(1, -1)

    available_books = len(book_pivot)
    neighbors_count = min(number_of_books + 1, available_books)

    distances, indices = model.kneighbors(
        book_vector,
        n_neighbors=neighbors_count,
    )

    recommendations = []

    for distance, recommended_index in zip(
        distances.flatten(),
        indices.flatten(),
    ):
        recommended_title = str(book_pivot.index[recommended_index])

        # Exclure le livre sélectionné
        if recommended_title == str(book_title):
            continue

        book_information = get_book_information(recommended_title)
        book_information["similarity"] = max(0, 1 - float(distance))

        recommendations.append(book_information)

        if len(recommendations) >= number_of_books:
            break

    return recommendations


def display_book(book: dict) -> None:
    if book.get("image"):
        st.image(
            book["image"],
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div style="
                height: 270px;
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #f1f3f5;
                border-radius: 12px;
                font-size: 70px;
            ">
                📖
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f"**{book['title']}**")

    if book.get("author"):
        st.caption(f"Par {book['author']}")

    


# ---------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------
st.title("📚 Book Recommender System")
st.write(
    "Sélectionnez un livre pour découvrir des ouvrages similaires."
)

available_book_names = sorted(
    {
        str(book_name).strip()
        for book_name in book_names
        if str(book_name).strip()
    }
)

selected_book = st.selectbox(
    "Recherchez ou sélectionnez un livre",
    options=available_book_names,
    index=None,
    placeholder="Saisissez le titre d’un livre...",
)

number_of_recommendations = st.slider(
    "Nombre de recommandations",
    min_value=1,
    max_value=10,
    value=5,
)

if st.button(
    "Obtenir des recommandations",
    type="primary",
    use_container_width=True,
):
    if not selected_book:
        st.warning("Veuillez sélectionner un livre.")
    else:
        try:
            with st.spinner("Recherche des livres similaires..."):
                recommendations = recommend_books(
                    selected_book,
                    number_of_recommendations,
                )

            if not recommendations:
                st.info(
                    "Aucune recommandation trouvée pour ce livre."
                )
            else:
                st.subheader(f"Parce que vous avez choisi : {selected_book}")

                selected_information = get_book_information(selected_book)

                selected_column, information_column = st.columns([1, 4])

                with selected_column:
                    display_book(selected_information)

                with information_column:
                    st.write(
                        "Voici les livres les plus proches selon "
                        "les préférences des lecteurs."
                    )

                st.divider()
                st.subheader("Livres recommandés")

                columns_per_row = min(5, len(recommendations))

                for start in range(
                    0,
                    len(recommendations),
                    columns_per_row,
                ):
                    columns = st.columns(columns_per_row)
                    row_books = recommendations[
                        start:start + columns_per_row
                    ]

                    for column, book in zip(columns, row_books):
                        with column:
                            display_book(book)

        except Exception as error:
            st.error(
                "Une erreur est survenue pendant la recommandation."
            )
            st.exception(error)
