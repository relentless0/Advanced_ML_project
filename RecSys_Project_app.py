import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('new_dataset.csv')  # Укажите путь к вашему датасету

def tokenize_track_name(track_name):
    fixed = track_name.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
    tokens = word_tokenize(fixed)
    return tokens

df['tokenized'] = df['track_name'].apply(tokenize_track_name)
model = Word2Vec(sentences=df['tokenized'], vector_size=100, window=5, min_count=1, workers=4)

numerical_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Установка фона
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/free-photo/beautiful-feathers-arrangement_23-2151436571.jpg');
        background-size: cover;
        background-position: center;
        color: white;  /* Цвет текста */
    }
    .css-1emrehy.edgvbvh3 {
        background-color: rgba(255, 255, 255, 0.8);
        color: black;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_combined_recommendations(track, artist, df, model, top_n=5):
    track_normalized = track.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
    artist_normalized = artist.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')

    if track_normalized not in model.wv:
        print('Трек не найден в модели.')
        return []

    by_name = model.wv.most_similar(track_normalized, topn=top_n)
    by_name = [(similar[0], round(similar[1], 3)) for similar in by_name]

    df['track_name_normalized'] = df['track_name'].str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace("'", '')
    df['artist_name_normalized'] = df['artist_name'].str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace("'", '')

    track_index = df[(df['track_name_normalized'] == track_normalized) & (df['artist_name_normalized'] == artist_normalized)].index
    if len(track_index) == 0:
        print(f"Трек '{track}' от '{artist}' не найден в датасете.")
        return []

    track_index = track_index[0]
    target_features = df.loc[track_index, numerical_features].values.reshape(1, -1)
    by_features = cosine_similarity(target_features, df[numerical_features]) 
    similar_indices = by_features[0].argsort()[-top_n-1:-1][::-1]    
    by_features = [(df.iloc[i]['track_name'], round(by_features[0][i], 3)) for i in similar_indices]

    combined_recommendations = {name: similarity for name, similarity in by_name} 
    for name, similarity in by_features:
        if name not in combined_recommendations:
            combined_recommendations[name] = similarity

    genre_weight = 5 
    popularity_weight = 1.3 
    artist_weight = 10  
    
    final_scores = {}
    for name, sim in combined_recommendations.items():
        genre_score = 0
        popularity_score = 0
        artist_score = 0
        
        genre_row = df[df['track_name'] == name] 
        if not genre_row.empty:
            genre_score = genre_row['genre'].values[0] 
            genre_score = 1 if genre_score in df.loc[track_index, 'genre'] else 0  
        
        popularity_row = df[df['track_name'] == name] 
        if not popularity_row.empty:
            popularity_score = popularity_row['popularity'].values[0]  

        artist_row = df[df['track_name'] == name] 
        if not artist_row.empty:
            artist_score = artist_weight if artist_row['artist_name_normalized'].values[0] == artist_normalized else 0
        
        final_score = (sim * 0.5 + 
                       (genre_weight * genre_score) + 
                       (popularity_weight * popularity_score) + 
                       artist_score)
        final_scores[name] = min(final_score, 100)  

    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    formatted_recommendations = []
    for name, score in sorted_recommendations[:top_n]:
        artist_name = df[df['track_name'] == name]['artist_name'].values[0]  
        formatted_recommendations.append((name, artist_name, round(score, 1))) 

    return formatted_recommendations

# Заголовок приложения
st.title('Система рекомендаций треков')

col1, col2 = st.columns(2)
with col1:
    track = st.text_input('Введите название трека:', key="track_input")
with col2:
    artist = st.text_input('Введите имя артиста:', key="artist_input")

# Инициализация переменной recommendations
recommendations = []

if st.button('Получить рекомендации'):
    recommendations = get_combined_recommendations(track, artist, df, model, top_n=10)
    
    if recommendations:
        st.write('**Рекомендованные треки:**')
        
        # Создаем DataFrame для красивого отображения
        rec_df = pd.DataFrame(recommendations, columns=['Трек', 'Артист', 'Оценка'])

        # Стилизация таблицы
        styled_rec_df = rec_df.style.applymap(
            lambda x: 'background-color: rgba(255, 255, 255, 0.5); color: black;' if isinstance(x, (int, float)) else '',
            subset=['Оценка']
        ).set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"') \
         .set_table_styles(
              [
                {'selector': 'th', 'props': [('border', '1px solid black'), ('background-color', 'rgba(255, 255, 255, 0.8)'), ('color', 'black')]},
                {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '8px')]}
            ]
        )

        st.dataframe(styled_rec_df)  # Используем st.dataframe для отображения стилизованного DataFrame
    else:
        st.write('Рекомендации не найдены.')

# Определение функций для метрик
def diversity_score(original_features):
    """Расчет диверсификации между оригинальными и рекомендованными треками."""
    distance_matrix = cosine_distances(original_features)
    diversity = 1 - distance_matrix.mean()
    return diversity

def mean_distance_score(original_track, recommended_features):
    """Расчет среднего расстояния между оригинальным треком и рекомендованными треками."""
    distances = cosine_distances([original_track], recommended_features)
    mean_distance = distances.mean()
    return mean_distance

# Проверка метрик
if st.button('Показать метрики'):
    if recommendations:
        recommended_names = [rec[0] for rec in recommendations]
        original_track_features = df[df['track_name'].isin(recommended_names)][numerical_features].values
        
        if original_track_features.size > 0:
            original_track = df[df['track_name'] == track][numerical_features].values[0]  
            diversity = diversity_score(original_track_features)
            mean_distance = mean_distance_score(original_track, original_track_features)

            st.write(f'**Диверсификация:** {diversity:.2f}')
            st.write(f'**Среднее расстояние:** {mean_distance:.2f}')

            # Построение графиков
            fig_diversity = px.line(x=recommended_names, y=[diversity]*len(recommended_names),
                                    labels={'x': 'Рекомендованные треки', 'y': 'Диверсификация'},
                                    title="График диверсификации рекомендаций")
            st.plotly_chart(fig_diversity)

            fig_mean_dist = px.histogram(x=recommended_names, y=[mean_distance]*len(recommended_names),
                                         labels={'x': 'Рекомендованные треки', 'y': 'Среднее расстояние'},
                                         title="Гистограмма среднего расстояния рекомендаций")
            st.plotly_chart(fig_mean_dist)
        else:
            st.write('Не удалось извлечь векторы признаков для рекомендованных треков.')
    else:
        st.write('Сначала получите рекомендации, чтобы увидеть метрики.')
