import pandas as pd
from keras.models import load_model
import os

def predict_random_spam(data, model):
    random_sample = data.sample(n=1)
    features = random_sample.drop('label', axis=1)
    label = random_sample['label'].iloc[0]
    prediction = model.predict(features, verbose=0)
    predicted_label = "Спам" if prediction[0][0] > 0.5 else "Не спам"
    actual_label = "Спам" if label == 1 else "Не спам"
    return f'Реальная метка: {actual_label}, Предсказанная метка: {predicted_label}'

# Выполнение предсказания на случайном примере из датасета
def predict_spam():
    current_path = os.getcwd()
    file_path = os.path.join(current_path, 'spambase.data')

    columns = [
        "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over",
        "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail", "word_freq_receive",
        "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
        "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", "word_freq_your",
        "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george",
        "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data",
        "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
        "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
        "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
        "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
        "capital_run_length_longest", "capital_run_length_total", "label"
    ]
    data = pd.read_csv(file_path, header=None, names=columns)

    model = load_model('model_linear.keras')
    return predict_random_spam(data, model)

# print(predict_spam())