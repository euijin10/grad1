
"""
A movie recommender app built with Streamlit.
"""

import numpy as np
import pandas as pd
import streamlit as st
import validators

from recommender import Recommender


@st.cache_data
def load_movies() -> pd.DataFrame:
    """
    Function to load prepared data from CSV files.
    """
    movies = pd.read_csv("./data/movies_imdb.csv")
    return movies


@st.cache_data
def get_random_movies_to_rate(num_movies: int = 5) -> pd.DataFrame:
    """
    Function to randomly get movie titles and ids to be rated by the user.
    """
    movies = load_movies()

    movies = movies.sort_values("imdb_rating", ascending=False).reset_index(drop=True)
    movies = movies[:100]

    select = np.random.choice(movies.index, size=num_movies, replace=False)

    return movies.iloc[select]


@st.cache_data
def get_movies() -> pd.DataFrame:
    """
    Function to get movie titles and ids to be selected by the user.
    """
    movies = load_movies()
    movies = movies.sort_values("title").reset_index(drop=True)

    return movies


@st.cache_data
def get_movie_id_from_title(title_str: str) -> int:
    """
    Function that returns a movies ID from a title input.
    """
    movies = load_movies()
    movies = movies[movies["title"] == title_str]["movie_id"]

    return int(movies.iloc[0])


def prepare_query_favourites() -> dict:
    """
    Funpiction to prepare query to search for movies based on favourite movies.
    """
    data = get_movies()

    st.markdown(
        "어떤 영화를 볼지 고민 중이신가요?"
        " **최애 영화 알려주시면** 취향에 맞게 추천해 드릴게요"
    )

    user_ratings = st.multiselect(
        "원하시는 영화 다 선택해 주세요.",
        data["title"],
    )

    query = {}
    for title_selected in user_ratings:
        # Get movie ids
        mid = get_movie_id_from_title(title_selected)
        # Set rating to 5 for selected movies
        query[mid] = 5

    return query


def prepare_query_rating() -> dict:
    """
    Function to prepare query to search for movies based on rating.
    """
    data = get_random_movies_to_rate(10)

    st.markdown(
        "어떤 영화를 봐야 할지 모르겠나요? 다음 무작위로 선태된 10편의 영화입니다."
        " **별점**을 매겨주시면 그 별점에 따라"
        "저희가 그 별점을 바탕으로 추천해 드리겠습니다."
    )

    query = {}
    for movie_id, title in zip(data["movie_id"], data["title"]):
        query[movie_id] = st.select_slider(title, options=[0, 1, 2, 3, 4, 5])

    return query


def recommender(rec_type: str = "fav") -> None:
    """
    Function to recommend movies.
    """

    # Prepare query based on type
    query = (
        prepare_query_rating() if rec_type == "rating" else prepare_query_favourites()
    )

    # Show select list for algorithm to use
    method_select = st.selectbox(
        "원하시는 알고리즘 선택해 주세요",
        ["Nearest Neighbors", "Non-negative matrix factorization"],
        key="method_selector_" + rec_type,
    )

    # Translate selection into keywords
    method = "neighbors" if method_select == "Nearest Neighbors" else "nmf"

    num_movies = st.slider(
        "몇 개의 영화를 추천해 드릴까요?",
        min_value=1,
        max_value=10,
        value=5,
        key="num_movies_slider_" + rec_type,
    )

    # Start recommender
    if st.button("영화 추천해 주세요!", key="button_" + rec_type):
        with st.spinner(f"Calculating recommendations using {method_select}..."):
            recommend = Recommender(query, method=method, k=num_movies)
            movie_ids, _ = recommend.recommend()

        with st.spinner("Fetching movie information from IMDB..."):
            st.write("Recommended movies using Nearest Neighbors:\n")
            for movie_id in movie_ids:
                display_movie(movie_id)


def display_movie(movie_id: int) -> None:
    """
    Function that displays a movie with information from IMDB.
    """
    movies = load_movies()
    movie = movies[movies["movie_id"] == movie_id].copy()

    col1, col2 = st.columns([1, 4])

    with col1:
        if validators.url(str(movie["cover_url"].iloc[0])):
            st.image(movie["cover_url"].iloc[0])

    with col2:
        if not pd.isnull(movie["title"].iloc[0]) and not pd.isnull(
            movie["year"].iloc[0]
        ):
            st.header(f"{movie['title'].iloc[0]} ({movie['year'].iloc[0]})")
        if not pd.isnull(movie["imdb_rating"].iloc[0]):
            st.markdown(f"**IMDB-rating:** {movie['imdb_rating'].iloc[0]}/10")
        if not pd.isnull(movie["genre"].iloc[0]):
            st.markdown(f"**Genres:** {', '.join(movie['genre'].iloc[0].split(' | '))}")
        if not pd.isnull(movie["director"].iloc[0]):
            st.markdown(
                f"**Director(s):** {', '.join(movie['director'].iloc[0].split('|'))}"
            )
        if not pd.isnull(movie["cast"].iloc[0]):
            st.markdown(
                f"**Cast:** {', '.join(movie['cast'].iloc[0].split('|')[:10])}, ..."
            )
        if not pd.isnull(movie["plot"].iloc[0]):
            st.markdown(f"{movie['plot'].iloc[0]}")
        if validators.url(str(movie["url"].iloc[0])):
            st.markdown(f"[Read more on imdb.com]({movie['url'].iloc[0]})")
    st.divider()


# Set page title
st.set_page_config(page_title="오늘 무슨 영화 볼까요?")

# Header image
st.image("data/cover_collage.jpg")

# Print title and subtitle
st.title("오늘 무슨 영화 볼까요?")
st.subheader("개인 영화 추천 시스템")

tab1, tab2 = st.tabs(["좋아하는 영화", "별점"])

with tab1:
    recommender("fav")

with tab2:
    recommender("rating")
