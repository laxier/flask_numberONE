from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bootstrap import Bootstrap4
from flask_moment import Moment
from flask_socketio import SocketIO
import os

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'))
app.config.from_object(Config)

socketio = SocketIO(app)

db = SQLAlchemy(app)

bootstrap = Bootstrap4(app)

moment = Moment(app)

login = LoginManager(app)
login.login_view = 'login'

from app import routes, models
