import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Загрузка сохраненной модели LightGBM
lgb_model = lgb.Booster(model_file='lgb_model1.txt')

# Загрузка сохраненного объекта MinMaxScaler
scaler = joblib.load('scaler_lgb.save')

# Предположим, что в data уже есть столбцы 'gender' и 'platform'
data=pd.read_csv('dummy_data.csv')
# Выбор случайного примера из data и масштабирование признаков
random_index = np.random.randint(0, len(data))

data=pd.read_csv('dummy_data.csv')
data_new = data[['age', 'gender', 'time_spent', 'platform']]
pd.set_option('display.max_columns', 50)
# Предобработка данных, например, кодирование категориальных переменных
data_processed = pd.get_dummies(data_new, columns=['gender', 'platform'])

data_processed_new = data_processed[['age', 'gender_female','time_spent', 'gender_male','gender_non-binary','platform_Facebook', 'platform_Instagram', 'platform_YouTube']]

random_data_processed = data_processed_new.iloc[random_index]
X_random = random_data_processed.drop('time_spent').to_frame().T
X_random_scaled = scaler.transform(X_random)

# Прогнозирование на случайном примере с помощью загруженной модели
y_random_pred = lgb_model.predict(X_random_scaled)

print("Случайный пример из data:")
print(random_data_processed)
print("Прогнозное значение времени на основе модели:", y_random_pred[0])
