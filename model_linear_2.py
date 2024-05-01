import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import joblib
from sklearn.metrics import r2_score
from tensorflow.keras.metrics import MeanAbsoluteError
# Загрузка данных
stock = 'BTC-USD'
start = '2014-09-17'
end = '2024-02-13'
df = yf.download(stock, start, end)
df.fillna(method='ffill', inplace=True)  # Заполнение пропущенных значений

# Нормализация столбца Close
scaler = MinMaxScaler(feature_range=(0,1))
scaled_close = scaler.fit_transform(df[['Close']])

# Функция для создания набора данных
def create_dataset(dataset, look_back=100):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Разделение данных на обучающие и тестовые
look_back = 100
train_size = int(len(scaled_close) * 0.8)
test_size = len(scaled_close) - train_size
train, test = scaled_close[0:train_size,:], scaled_close[train_size-look_back:len(scaled_close),:]

# Вызов функции create_dataset для обучающего и тестового наборов
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# Изменение формы для LSTM
x_train = np.reshape(x_train, (x_train.shape[0], look_back, 1))
x_test = np.reshape(x_test, (x_test.shape[0], look_back, 1))

# Создание LSTM модели
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    LSTM(60, return_sequences=False),
    Dropout(0.3),
    Dense(1, activation='linear')  # Выходной слой с линейной активацией
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])

# Обучение модели с EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=1)

# Предсказания
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Обратное масштабирование
# model.save("model_linear_2.keras")
# # Визуализация
# plt.figure(figsize=(10,6))
# plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)), label='Actual Price', color="orange")
# plt.plot(predictions, label='Predicted Price', color='blue')
# plt.title('Bitcoin Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

# Сохранение модели и scaler
model.save("model_linear_2.keras")
joblib.dump(scaler, 'btc_price_scaler.pkl')
r2 = r2_score(y_test, predictions)
loss, mse = model.evaluate(x_test, y_test, verbose=1)
print(f'Test MSE: {mse}')
print(f'R²: {r2}')
# epsilon = 1e-10
# percentage_change = (y_predict_rescaled - y_test_rescaled) / (y_test_rescaled + epsilon)
#
# # # Определение сигналов к покупке и продаже
# buy_signals = percentage_change > 0.01  # Сигнал к покупке, если предсказанная цена на 1% выше
# sell_signals = percentage_change < -0.01  # Сигнал к продаже, если предсказанная цена на 1% ниже
# #
# # Визуализация результатов
# plt.figure(figsize=(14, 7))
# plt.plot(y_test_rescaled, color='g', label='Original Prices')
# plt.plot(y_predict_rescaled, color='r', linestyle='--', label='Predicted Price')
# buy_indices = np.nonzero(buy_signals)[0]
# sell_indices = np.nonzero(sell_signals)[0]
# plt.scatter(np.where(buy_signals), y_test_rescaled[buy_signals], marker='^', color='b', label='Buy Signal', alpha=1)
# plt.scatter(np.where(sell_signals), y_test_rescaled[sell_signals], marker='v', color='m', label='Sell Signal', alpha=1)
# plt.title('Stock Price Prediction with Buy & Sell Signals')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

