import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
from collections import defaultdict
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to load batting data
def load_data():
    data = pd.read_csv("example_data/game_batting_logs.csv")  # Adjust path as necessary
    data['Year'] = data['Year'].astype(int)
    return data.groupby(['Name', 'Year'])['H'].mean().reset_index()

# Predict future hits for all players
def predict_hits_for_all(annual_hits):
    predictions = {}
    for player in annual_hits['Name'].unique():
        player_data = annual_hits[annual_hits['Name'] == player]
        if len(player_data) > 2:  # Ensuring sufficient data points
            prediction = predict_hits(player_data)
            predictions[player] = prediction
    return predictions

# Predict future hits for a single player
def predict_hits(player_data):
    X = player_data[['Year']]
    y = player_data['H']
    model = LinearRegression()
    model.fit(X, y)
    future_year = np.array([[2024]])
    prediction = model.predict(future_year)
    return np.clip(prediction, a_min=0, a_max=None)[0]

# Function to parse articles and calculate sentiment scores
def parse_articles_for_sentiment(file_path, players):
    with open(file_path, 'r') as file:
        articles = json.load(file)
    sentiment_scores = defaultdict(float)
    max_length = 512  # Typical max length for transformer models

    for article in articles:
        for player in players:
            if player in article:
                # Splitting the article into chunks that respect the model's maximum input length
                parts = [article[i:i+max_length] for i in range(0, len(article), max_length)]
                total_score = 0
                for part in parts:
                    sentiment_result = sentiment_pipeline(part)
                    for result in sentiment_result:
                        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                        total_score += score
                sentiment_scores[player] += total_score / len(parts)  # Averaging the score across parts
    return sentiment_scores

# Main function to run Streamlit app
def main():
    st.title("Baseball Players' Performance and Sentiment Analysis")
    
    # Load and process data
    annual_hits = load_data()
    players = annual_hits['Name'].unique()
    
    # Player selection
    selected_player = st.selectbox("Select a player", players)
    
    # Predictions for selected player
    selected_player_data = annual_hits[annual_hits['Name'] == selected_player]
    if len(selected_player_data) > 2:
        st.subheader(f"Predicted Hits for {selected_player} in 2024")
        prediction = predict_hits(selected_player_data)
        st.write(f"Predicted hits for {selected_player} in 2024: {prediction}")

        # Linear regression model for selected player
        X = selected_player_data[['Year']]
        y = selected_player_data['H']
        model = LinearRegression()
        model.fit(X, y)

        # Plot linear regression graph
        fig, ax = plt.subplots()
        ax.scatter(selected_player_data['Year'], selected_player_data['H'], color='blue', label='Actual Hits')
        ax.set_xlabel('Year')
        ax.set_ylabel('Hits')
        ax.plot(selected_player_data['Year'], model.predict(X), color='red', linewidth=2, label='Linear Regression')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write(f"Not enough data available for {selected_player} to make predictions.")

    # Sentiment analysis for all players
    st.subheader("Sentiment Analysis for All Players")
    sentiments = parse_articles_for_sentiment("example_data/articles.json", players)
    sentiment_df = pd.DataFrame(list(sentiments.items()), columns=['Player', 'Sentiment']).sort_values('Sentiment', ascending=False)
    fig, ax = plt.subplots(figsize=(8, len(sentiments)/2))
    ax.barh(sentiment_df['Player'], sentiment_df['Sentiment'], color='lightgreen')
    ax.invert_yaxis()  # Display the highest value at the top
    ax.set_xlabel('Sentiment Score')
    ax.set_title('Player Sentiment Based on Article Analysis')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
