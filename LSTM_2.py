import yfinance as yf
from finta import TA
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from tensorflow.keras.metrics import MeanAbsoluteError
import joblib
import os


def study_lstm(epos):
    try:
        # Загрузка данных
        stock = 'BTC-USD'
        start = '2014-09-17'
        end = '2024-02-13'

        df = yf.download(stock, start, end)
        df.index = df.index.date
        df.fillna(0, inplace=True)
        df['RSI'] = TA.RSI(df, 12)
        df['SMA'] = TA.SMA(df)
        df['OBV'] = TA.OBV(df)
        df = df.fillna(0)
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        test_data = df['Close'][int(0.8*len(df))-100:].values
        scaler_test = MinMaxScaler()
        scaled_data = scaler_test.fit_transform(test_data.reshape(-1, 1))

        x_test, y_test = [], []

        for i in range(100, len(test_data)):
            x_test.append(scaled_data[i-100:i, 0])
            y_test.append(scaled_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # Подготовка тренировочных данных
        training_data = df['Close'][:int(0.8*len(df))].values
        training_data = scaler.fit_transform(training_data.reshape(-1, 1))

        x_train, y_train = [], []

        for i in range(100, len(training_data)):
            x_train.append(training_data[i-100:i, 0])
            y_train.append(training_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Преобразование для LSTM

        # Создание и компиляция модели
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=60, return_sequences=True),
            Dropout(0.3),
            LSTM(units=80, return_sequences=True),
            Dropout(0.4),
            LSTM(units=120, return_sequences=False),
            Dropout(0.5),
            Dense(units=1, activation='linear')
        ])



        # Обучение модели
        # model.compile(optimizer='adam', loss='mean_squared_error')
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])

        # Определение функции обратного вызова для отправки прогресса обучения через сокеты
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        # Обучение модели с использованием функции обратного вызова SocketCallback
        history = model.fit(x_train, y_train, epochs=epos, batch_size=64, verbose=0, validation_data=(x_test, y_test),
                            callbacks=[early_stopping])
        y_predict = model.predict(x_test, verbose=0)
        model.save("model_linear0.keras")
        # close_prices = df[['Close']].values
        # scaler = MinMaxScaler(feature_range=(0,1))
        # scaler.fit(close_prices)  # close_prices - это ваш массив данных для обучения scaler
        # joblib.dump(scaler, 'scaler_fit.save')
        # joblib.dump(scaler, 'btc_price_scaler.pkl')

        # #для R^2
        r2 = r2_score(y_test, y_predict)
        loss, mse = model.evaluate(x_test, y_test, verbose=0)
        # print(f'Test MSE: {mse}')
        # print(f'R²: {r2}')
        return f'Test MSE: {mse} R^2: {r2}'
    except Exception as e:
        return str(e)
    # scaler_close = MinMaxScaler()
    # df['Close'] = scaler_close.fit_transform(df[['Close']])
    # y_predict_rescaled = scaler_close.inverse_transform(y_predict)
    # y_test_rescaled = scaler_close.inverse_transform(y_test.reshape(-1, 1))