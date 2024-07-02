from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and prepare data
def load_and_prepare_data():
    data = pd.read_csv('/mnt/data/data.csv')
    data['Dates'] = pd.to_datetime(data['Dates'], format='%d-%b')
    data.set_index('Dates', inplace=True)
    rainfall_data = data[['ACTUAL']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rainfall_data['ACTUAL'] = scaler.fit_transform(rainfall_data[['ACTUAL']])
    return rainfall_data, scaler

rainfall_data, scaler = load_and_prepare_data()

# Create sequences
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

sequence_length = 30
X, y = create_sequences(rainfall_data['ACTUAL'].values, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['data']
    data = np.array([float(x) for x in data.split(',')]).reshape(-1, 1)
    data = scaler.transform(data)
    X_new = np.array([data[-sequence_length:]])
    X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
    prediction = model.predict(X_new)
    prediction = scaler.inverse_transform(prediction)
    
    # Plot the results
    img = io.BytesIO()
    plt.figure(figsize=(14, 5))
    plt.plot(range(sequence_length), scaler.inverse_transform(data[-sequence_length:]), label='Input Rainfall')
    plt.plot(sequence_length, prediction, 'ro', label='Predicted Fluid Level')
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', prediction=prediction[0][0], plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
