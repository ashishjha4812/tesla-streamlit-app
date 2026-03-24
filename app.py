
#import libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#load dataset

df = pd.read_csv(r'C:\Users\LENEVO\PyCharmMiscProject\TSLA.csv')

df.head()

#basic info
df.info()
df.isnull().sum()

#Data preprocessing
#convert date and set index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#use closing price
data = df[['Close']]

#visualization
plt.figure(figsize=(12,6))
plt.plot(data, label='Closing Price')
plt.title('Tesla Stock Price')
plt.legend()
plt.show()

#scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

#create time series sequence
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

#reshape from RNN to LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

#train test split
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#simple RNN model
rnn_model = Sequential()

rnn_model.add(SimpleRNN(50, return_sequences=True, input_shape=(60,1)))
rnn_model.add(Dropout(0.2))

rnn_model.add(SimpleRNN(50))
rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(1))

rnn_model.compile(optimizer='adam', loss='mean_squared_error')

rnn_model.summary()

#train
early_stop = EarlyStopping(monitor='val_loss', patience=5)

rnn_model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=20,
              batch_size=32,
              callbacks=[early_stop])

#LSTM model
lstm_model = Sequential()

lstm_model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(50))
lstm_model.add(Dropout(0.2))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.summary()

#train
lstm_model.fit(X_train, y_train,
               validation_data=(X_test, y_test),
               epochs=20,
               batch_size=32,
               callbacks=[early_stop])

#prediction
rnn_pred = rnn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)

# Inverse scaling
rnn_pred = scaler.inverse_transform(rnn_pred)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))


#evaluation
rnn_mse = mean_squared_error(y_test_actual, rnn_pred)
lstm_mse = mean_squared_error(y_test_actual, lstm_pred)

print("SimpleRNN MSE:", rnn_mse)
print("LSTM MSE:", lstm_mse)

#visualization

plt.figure(figsize=(12,6))

plt.plot(y_test_actual, label='Actual')
plt.plot(rnn_pred, label='RNN Predictions')
plt.plot(lstm_pred, label='LSTM Predictions')

plt.legend()
plt.title('Stock Price Prediction Comparison')
plt.show()

#predict future(1,5,10) days
def predict_future(model, data, days=10):
    temp_input = data[-60:].reshape(1,60,1)
    predictions = []

    for _ in range(days):
        pred = model.predict(temp_input)[0][0]
        predictions.append(pred)

        temp_input = np.append(temp_input[:,1:,:], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# Example
future_1 = predict_future(lstm_model, scaled_data, 1)
future_5 = predict_future(lstm_model, scaled_data, 5)
future_10 = predict_future(lstm_model, scaled_data, 10)

print("1 Day Prediction:", future_1)
print("5 Day Prediction:", future_5)
print("10 Day Prediction:", future_10)


#Save Model & Scaler
import pickle

# Save model
lstm_model.save("model.h5")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)



