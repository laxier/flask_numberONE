import yfinance as yf
import pandas as pd
from keras.models import load_model
import joblib

def predict_day_price(date, loaded_model, df, scaler, sequence_length=100):
    # Преобразование введенной даты в формат datetime
    # date = pd.to_datetime(date_str)

    # Находим индекс даты в DataFrame
    date_index = df.index.get_loc(date)

    # Выборка последовательности цен закрытия за последние sequence_length дней перед указанной датой
    start_index = date_index - sequence_length
    end_index = date_index
    last_sequence = df['Close'][start_index:end_index].values

    # Нормализация последовательности
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

    # Изменение формы последовательности для предсказания
    last_sequence_scaled = last_sequence_scaled.reshape((1, sequence_length, 1))

    # Предсказание цены на следующий день
    predicted_price_scaled = loaded_model.predict(last_sequence_scaled, verbose=0)

    # Обратное масштабирование предсказанной цены
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0][0]

def get_real_pericted(start_date, end_date):
    stock = 'BTC-USD'
    start = '2014-09-17'
    end = '2024-03-07'
    df = yf.download(stock, start, end)
    # df = df.to_csv().encode('utf-8')

    df.index = pd.to_datetime(df.index)
    df.fillna(method='ffill', inplace=True)
    loaded_model = load_model('D:\\flaskProject\\model_linear.keras')
    scaler = joblib.load('D:\\flaskProject\\btc_price_scaler.pkl')

    # Получение данных за год 2015
    df_time = df.loc[start_date:end_date]
    # Создание списков для хранения предсказанных цен и реальных цен закрытия за 2015 год
    predicted_prices = []
    real_prices = []

    for date in df_time.index:
        predicted_price = predict_day_price(date, loaded_model, df, scaler)
        predicted_prices.append(predicted_price)

        real_price = df.loc[df.index == date]['Close'].values[0]
        real_prices.append(real_price)

    df_results = pd.DataFrame({'Date': df_time.index, 'Real Price': real_prices, 'Predicted Price': predicted_prices}).reset_index(drop=True)
    return df_results