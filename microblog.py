import sqlalchemy as sa
import sqlalchemy.orm as so
from app import app, db
from app.models import User, Post

@app.shell_context_processor
def make_shell_context():
    return {'sa': sa, 'so': so, 'db': db, 'User': User, 'Post': Post}

if __name__ == "__main__":
    # with app.app_context():
    #     db.create_all()
    #     db.session.add(User(username='admin',
    #                         password = generate_password_hash('12345678'),
    #                         email="amdin@example.com"))
    #     db.session.commit()
    app.run()