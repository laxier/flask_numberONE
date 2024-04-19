from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import LoginForm
from app.forms import RegistrationForm
from app.forms import PostForm
from app.models import Post
from flask import request
from flask_login import current_user, login_user
from flask_login import logout_user
from flask_login import login_required
from urllib.parse import urlsplit
from werkzeug.utils import secure_filename
import sqlalchemy as sa
from app import db
from app.models import User

import os

@app.route('/', methods=["POST", "GET"])
@app.route('/index', methods=["POST", "GET"])
@login_required
def index():
    form = PostForm()
    if form.validate_on_submit():
        file_path = ""
        file = form.upload.data
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('Your file has been uploaded successfully!')
        post = Post(body=form.post.data, author=current_user, image_path=file_path)
        db.session.add(post)
        db.session.commit()
        flash('Your post is now live!')
        return redirect(url_for('index'))
    posts = Post.query.order_by(Post.timestamp).all()[::-1]
    return render_template("index.html", title='Home Page', form=form, posts=posts)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(sa.select(User).where(User.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/delete/<int:id>')
def delete_post(id):
    to_delete = Post.query.get_or_404(id)
    if to_delete.author.id == current_user.id or current_user.username == 'admin':
        try:
            file_path=to_delete.image_path
            if os.path.exists(file_path):
                os.remove(file_path)
            db.session.delete(to_delete)
            db.session.commit()
            return redirect('/')
        except:
            return "There was a problem"
    else:
        return "Action is not allowed"
    return render_template('register.html', title='Register', form=form)


@app.route("/info")
def info():
    return render_template('info.html')
