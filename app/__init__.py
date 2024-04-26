from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bootstrap import Bootstrap5
from flask_moment import Moment
import os

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'))
app.config.from_object(Config)

db = SQLAlchemy(app)

bootstrap = Bootstrap5(app)

moment = Moment(app)

login = LoginManager(app)
login.login_view = 'login'

from app import routes, models
