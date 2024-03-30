import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import tempfile

# Define the neural network architecture


class BiddingValueNet(nn.Module):
    def __init__(self, num_features):
        super(BiddingValueNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output layer for regression

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def load_data():
    # Load your data here, adjust paths as necessary
    free_agents = pd.read_csv('example_data/submission_example.csv')
    batting_logs = pd.read_csv('example_data/batting_season_summary.csv')
    return free_agents, batting_logs


def preprocess_data(batting_logs):
    features = batting_logs[['age', 'pos', 'PA', 'AB', '2B',
                             '3B', 'HR', 'BB', 'SO', 'P/PA', 'BA', 'OBP', 'SLG', 'OPS']]
    label = batting_logs['H']
    categorical_features = ['pos']
    numerical_features = ['age', 'PA', 'AB', '2B', '3B',
                          'HR', 'BB', 'SO', 'P/PA', 'BA', 'OBP', 'SLG', 'OPS']
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(
    ), numerical_features), ('cat', OneHotEncoder(), categorical_features)])
    X_train, X_val, y_train, y_val = train_test_split(
        features, label, test_size=0.2, random_state=42)
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    return X_train, X_val, y_train, y_val, preprocessor, features.columns


#@st.cache_data()
def train_model(X_train, y_train, num_features):
    model = BiddingValueNet(num_features)
    learning_rate = 0.001
    epochs = 500
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    X_train_torch = torch.tensor(X_train.astype(np.float32))
    y_train_torch = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_torch)
        loss = criterion(y_pred, y_train_torch)
        loss.backward()
        optimizer.step()
    return model


def predict_second_half(player_name, batting_logs, preprocessor, model, feature_columns):
    player_data = batting_logs[batting_logs['Name']
                               == player_name].sort_values(by='Year')

    if player_data.empty or len(player_data) < 2:
        st.write(f"Not enough data for {player_name} to make a prediction.")
        return

    # Finding the midpoint of the player's career for the "second half"
    mid_point = len(player_data) // 2
    first_half = player_data.iloc[:]
    second_half = player_data.iloc[mid_point:]

    # Preparing plot
    plt.figure(figsize=(10, 6))
    plt.plot(first_half['Year'], first_half['H'],
             label='First Half (Actual)', marker='o')

    predicted_hits = []
    for _, row in second_half.iterrows():
        current_year_data = row[feature_columns].to_frame().T
        processed_data = preprocessor.transform(current_year_data)
        data_tensor = torch.tensor(processed_data.astype(np.float32))
        model.eval()
        with torch.no_grad():
            predicted_hit = model(data_tensor).item()
        predicted_hits.append(predicted_hit)

    plt.plot(second_half['Year'], predicted_hits,
             label='Second Half (Predicted)', marker='x')
    plt.xlabel('Year')
    plt.ylabel('Hits')
    plt.title(f'Predicted vs Actual Hits for {player_name}')
    plt.legend()
    st.pyplot(plt)


def predict_next_year(player_name, batting_logs, preprocessor, model, feature_columns):
    player_data = batting_logs[batting_logs['Name'] == player_name]
    if player_data.empty or len(player_data) < 1:
        st.write(f"No sufficient data for {player_name} to make a prediction.")
        return
    last_season_stats = player_data.iloc[-1][feature_columns].to_frame().T
    processed_stats = preprocessor.transform(last_season_stats)
    stats_tensor = torch.tensor(processed_stats.astype(np.float32))
    model.eval()
    with torch.no_grad():
        predicted_hits = model(stats_tensor).item()
    return predicted_hits


def main():
    st.title("Baseball Player Hits Prediction")
    free_agents, batting_logs = load_data()

    if st.button("Train Model"):
        X_train, X_val, y_train, y_val, preprocessor, feature_columns = preprocess_data(
            batting_logs)
        model = train_model(X_train, y_train, X_train.shape[1])
        st.success("Model trained successfully!")

    else:
        model = None

    player_name = st.selectbox(
        "Select a Player to Predict Hits for Next Season", free_agents['Name'])
    if model and player_name:
        predicted_hits = predict_next_year(
            player_name, batting_logs, preprocessor, model, feature_columns)
        st.write(
            f"Predicted hits for {player_name} in the upcoming season: {predicted_hits:.0f}")
        predict_second_half(player_name, batting_logs,
                            preprocessor, model, feature_columns)


if __name__ == "__main__":
    main()
