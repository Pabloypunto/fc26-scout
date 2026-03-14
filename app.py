import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

st.title("⚽ EA FC Career Mode Scout")

st.write("Busca jugadores similares escribiendo su nombre.")

@st.cache_data
def load_data():
    df = pd.read_csv("players.csv")
    return df

players = load_data()

# columnas de atributos usadas para similitud
stats = [
"attributes.acceleration",
"attributes.sprintspeed",
"attributes.agility",
"attributes.balance",
"attributes.stamina",
"attributes.strength",
"attributes.reactions",
"attributes.vision",
"attributes.ballcontrol",
"attributes.dribbling",
"attributes.finishing",
"attributes.shortpassing",
"attributes.longpassing",
"attributes.shotpower",
"attributes.longshots",
"attributes.standingtackle",
"attributes.slidingtackle"
]

# lista de nombres de jugadores
player_names = players["info.name.knownas"].dropna().unique()

# barra de búsqueda con autocompletar
player_name = st.selectbox(
"Escribe o busca un jugador:",
player_names
)

if st.button("Buscar jugadores similares"):

    target = players[players["info.name.knownas"] == player_name]

    scaler = StandardScaler()
    scaled_players = scaler.fit_transform(players[stats])
    scaled_target = scaler.transform(target[stats])

    distances = euclidean_distances(scaled_players, scaled_target)

    players["similarity"] = distances

    results = players[
        players["info.name.knownas"] != player_name
    ].sort_values("similarity")

    st.subheader("Jugadores más similares")

    st.dataframe(
        results[[
            "info.name.knownas",
            "info.age",
            "info.overallrating",
            "primary_position",
            "info.valueEUR"
        ]].head(10)
    )