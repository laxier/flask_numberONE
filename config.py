import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    # SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:PolinaZh1301*@localhost/blog'
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:rootroot@localhost/blog'
    UPLOAD_FOLDER = 'static/images/'

