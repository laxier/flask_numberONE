import yfinance as yf
import pandas as pd
from keras.models import load_model
import joblib

def predict_next_day_price(date_str, loaded_model, df, scaler, sequence_length=100):
    # Преобразование введенной даты в формат datetime
    date = pd.to_datetime(date_str)

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
    predicted_price_scaled = loaded_model.predict(last_sequence_scaled)

    # Обратное масштабирование предсказанной цены
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0][0]



def predict_day_price(date_str, loaded_model, df, sequence_length=100):
    date = pd.to_datetime(date_str)
    date_index = df.index.get_loc(date)
    start_index = date_index - sequence_length
    end_index = date_index
    last_sequence = df['Close'][start_index:end_index].values
    last_sequence_reshaped = last_sequence.reshape((1, sequence_length, 1))
    predicted_price = loaded_model.predict(last_sequence_reshaped)
    return predicted_price[0][0]


def get_real_pericted(df, start_date, end_date):
    # Получение данных за год 2015
    df_time = df.loc[start_date:end_date]
    # Создание списков для хранения предсказанных цен и реальных цен закрытия за 2015 год
    predicted_prices = []
    real_prices = []

    for date in df_time.index:
        # predicted_price = predict_day_price(date.strftime('%Y-%m-%d'), loaded_model, df)
        predicted_price = predict_next_day_price(date.strftime('%Y-%m-%d'), loaded_model, df, scaler)
        predicted_prices.append(predicted_price)

        real_price = df.loc[df.index == date]['Close'].values[0]
        real_prices.append(real_price)

    df_results = pd.DataFrame({'Date': df_time.index, 'Real Price': real_prices, 'Predicted Price': predicted_prices})
    return df_results

# loaded_model = load_model('model_linear.keras')
loaded_model = load_model('model_linear.keras')
scaler = joblib.load('btc_price_scaler.pkl')
# scaler = joblib.load('scaler_fit.save')

stock = 'BTC-USD'
start = '2014-09-17'
end = '2024-03-07'
df = yf.download(stock, start, end)
df.index = pd.to_datetime(df.index)
df.fillna(method='ffill', inplace=True)
start_date = '2015-02-11'
end_date = '2015-03-1'

print(get_real_pericted(df, start_date, end_date))