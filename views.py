from django.contrib.auth import logout, login
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect, render
from django.views.generic import ListView, DetailView, TemplateView, CreateView
from django.urls import reverse_lazy
from django.conf import settings
import os

from django.db.models import Q
import lightgbm as lgb
from .models import Post, DbPolRegression, DbGradient
from .forms import RegisterUserForm, LoginUserForm

import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class BlogList(ListView):
    paginate_by = 3
    ordering = ["id"]
    model = Post
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RegisterUserForm
        context['form1'] = LoginUserForm
        return context

    def post(self, request, *args, **kwargs):
        if 'password1' in request.POST and 'email' in request.POST:
            return RegUser.as_view()(request)
        if 'password' in request.POST:
            return LogUser.as_view()(request)


class BlogDetailView(DetailView):
    model = Post
    template_name = 'post_detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RegisterUserForm
        context['form1'] = LoginUserForm
        return context

    def post(self, request, *args, **kwargs):
        if 'password1' in request.POST and 'email' in request.POST:
            return RegUser.as_view()(request)
        if 'password' in request.POST:
            return LogUser.as_view()(request)


class AboutPageView(TemplateView):
    template_name = 'about.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RegisterUserForm
        context['form1'] = LoginUserForm
        return context

    def post(self, request, *args, **kwargs):
        if 'password1' in request.POST and 'email' in request.POST:
            return RegUser.as_view()(request)
        if 'password' in request.POST:
            return LogUser.as_view()(request)


class SciencePageView(TemplateView):
    template_name = 'science.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RegisterUserForm
        context['form1'] = LoginUserForm
        return context

    def post(self, request, *args, **kwargs):
        if 'password1' in request.POST and 'email' in request.POST:
            return RegUser.as_view()(request)
        if 'password' in request.POST:
            return LogUser.as_view()(request)


class EntertainmentPageView(TemplateView):
    template_name = 'entertainment.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RegisterUserForm
        context['form1'] = LoginUserForm
        return context

    def post(self, request, *args, **kwargs):
        if 'password1' in request.POST and 'email' in request.POST:
            return RegUser.as_view()(request)
        if 'password' in request.POST:
            return LogUser.as_view()(request)


class NeanderthalPageView(TemplateView):
    template_name = 'neanderthal.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RegisterUserForm
        context['form1'] = LoginUserForm
        return context

    def post(self, request, *args, **kwargs):
        if 'password1' in request.POST and 'email' in request.POST:
            return RegUser.as_view()(request)
        if 'password' in request.POST:
            return LogUser.as_view()(request)


class RegUser(CreateView):
    form_class = RegisterUserForm

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('home')


class LogUser(LoginView):
    form_class = LoginUserForm

    def get_success_url(self):
        return reverse_lazy('home')


def logout_user(request):
    logout(request)
    return redirect('home')


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def LSTM_n(request):
    flight_data = sns.load_dataset("flights")
    print(flight_data.head())

    data_year = flight_data['year'].values.astype(int)[-12:]
    data_month = flight_data['month'].values.astype(str)[-12:]
    data_passengers = flight_data['passengers'].values.astype(float)

    test_data_size = 12
    train_data_pass = data_passengers[:-test_data_size]
    test_data_pass = data_passengers[-test_data_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data_pass.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_window = 12
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 300
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    fut_pred = 12
    test_inputs = train_data_normalized[-train_window:].tolist()
    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    predictions = scaler.inverse_transform(np.array(test_inputs[fut_pred:]).reshape(-1, 1))
    positions = []

    for i in range(fut_pred):
        positions.append(DbPolRegression(year=data_year[i], month=data_month[i], passengers=test_data_pass[i],
                                         predictions=predictions[i][0],
                                         wrong_answer=test_data_pass[i] - predictions[i][0]))

    DbPolRegression.objects.all().delete()
    DbPolRegression.objects.bulk_create(positions)
    return redirect('LSTM')


def Gradient(request):
    file_path = os.path.join(settings.BASE_DIR, 'web', 'dummy_data.csv')
    data = pd.read_csv(file_path)
    data_new = data[['age', 'gender', 'time_spent', 'platform']]
    pd.set_option('display.max_columns', 50)

    # Предобработка данных, например, кодирование категориальных переменных
    data_processed = pd.get_dummies(data_new, columns=['gender', 'platform'])
    print(data_processed)
    data_processed_new = data_processed[
        ['age', 'gender_female', 'time_spent', 'gender_male', 'gender_non-binary', 'platform_Facebook',
         'platform_Instagram', 'platform_YouTube']]

    # Разделение данных на предикторы (X) и целевую переменную (y)
    X = data_processed_new.drop('time_spent', axis=1)
    y = data['time_spent']

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание объекта Dataset для LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Параметры модели
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # Обучение модели
    num_round = 100
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=True)]
    lgb_model = lgb.train(params, train_data, num_boost_round=num_round, valid_sets=[test_data], callbacks=callbacks)

    # Прогнозирование на тестовом наборе
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    def calculate_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        return mape

    mape_score = calculate_mape(y_test, y_pred)
    print(f"MAPE: {mape_score:.2f}%")
    # Оценка качества модели
    mse = mean_squared_error(y_test, y_pred)
    lgb_model.save_model('lgb_model.txt')

    print('Mean Squared Error:', mse)

    DbGradient.objects.all().delete()
    DbGradient.objects.bulk_create(DbGradient(age=10, gender='male', platform='Facebook', y_pred=y_pred))
    return redirect('Gradient')


def Gradient_site_view(request):
    DB = DbGradient.objects.all()
    return render(request, "Gradient.html", {"DB2": DB})


def LSTM_site_view(request):
    DB = DbPolRegression.objects.all()
    return render(request, "LSTM.html", {"DB": DB})


class SearchResultsView(ListView):
    model = Post
    template_name = 'search.html'
    paginate_by = 3

    def get_queryset(self):
        query = self.request.GET.get('q')
        if len(query) != 0:
            object_list = Post.objects.filter(
                Q(title__contains=query) | Q(author__username__contains=query) | Q(body__contains=query)
            )
            print(object_list)
            return object_list
        else:
            return []


class BigSearchResultsView(ListView):
    model = Post
    template_name = 'big_search.html'

    def get_queryset(self):
        ob = list()
        query = self.request.GET.get('query')
        query_chbox = self.request.GET.getlist('query_chbox')
        if query is not None and '1' in query_chbox:
            for i in Post.objects.filter(Q(author__username__icontains=query)):
                ob.append(i.author.username)
        try:
            if query is not None and '2' in query_chbox:
                for i in Post.objects.filter(Q(author__date_joined=query)):
                    ob.append(i.author.date_joined)
        except Exception:
            pass
        if query is not None and '3' in query_chbox:
            for i in Post.objects.filter(Q(author__email__icontains=query)):
                ob.append(i.author.email)
        if query is not None and '5' in query_chbox:
            for i in Post.objects.filter(Q(title__icontains=query)):
                ob.append(i.title)
        if query is not None and '6' in query_chbox:
            for i in Post.objects.filter(Q(body__icontains=query)):
                ob.append(i.body)
        ob1 = list()
        for i in ob:
            if i not in ob1:
                ob1.append(i)
        return ob1
