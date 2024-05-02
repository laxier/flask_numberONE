from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import LoginForm, RegistrationForm
from app.forms import PostForm, CommentForm
from app.models import Post, Comment
from flask import request
from flask_login import current_user, login_user
from flask_login import logout_user
from flask_login import login_required
from urllib.parse import urlsplit
from werkzeug.utils import secure_filename
import sqlalchemy as sa
from app import db
from app.models import User

from LSTM_2 import study_lstm
from LSTM_2_predict import get_real_pericted

from RNN import study_rnn
from RNN_2 import predict_spam

from Gradient_random import grad_random
from Gradient_test import study_grad
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
            upload_folder = app.config['UPLOAD_FOLDER']

            # Check if the file already exists in the destination folder
            if os.path.exists(os.path.join(upload_folder, filename)):
                # If the file already exists, create a new filename
                filename, file_extension = os.path.splitext(filename)
                new_filename = f"{filename}_1{file_extension}"
                # Continue this loop to find a unique filename
                i = 1
                while os.path.exists(os.path.join(upload_folder, new_filename)):
                    i += 1
                    new_filename = f"{filename}_{i}{file_extension}"

                # Save the file with the new unique filename
                file_path = os.path.join(upload_folder, new_filename)
            else:
                file_path = os.path.join(upload_folder, filename)
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
        db.session.delete(to_delete)
        db.session.commit()
        try:
            db.session.delete(to_delete)
            db.session.commit()

            file_path = to_delete.image_path
            if os.path.exists(file_path):
                os.remove(file_path)
            flash('Post deleted')
            return redirect('/')
        except:
            return "There was a problem"
    else:
        return "Action is not allowed"
    return redirect(url_for('index'))


@app.route('/post/<int:commID>/delete/<int:id>')
def delete_comm(commID, id):
    to_delete = Comment.query.get_or_404(id)
    if to_delete.author.id == current_user.id or current_user.username == 'admin':
        try:
            db.session.delete(to_delete)
            db.session.commit()
            flash('Comment deleted')
            return redirect(f'/post/{commID}')
        except:
            return "There was a problem"
    else:
        return "Action is not allowed"
    return redirect(f'/post/{id}')


@app.route('/post/<int:id>', methods=['GET', 'POST'])
def comment_post(id):
    form = CommentForm()
    post = Post.query.get_or_404(id)
    comments = Comment.query.filter_by(post_id=id).order_by(Comment.timestamp).all()[::-1]
    if form.validate_on_submit():
        comment = Comment(body=form.post.data,
                          post=post,
                          author=current_user)
        db.session.add(comment)
        db.session.commit()
        flash('You have commented on the post')
        return redirect(f'/post/{id}')
    return render_template('post.html', title=f'Пост {id}', form=form, post=post, comments=comments)


@app.route("/info")
def info():
    return render_template('info.html')

@app.route("/neuro/lstm")
def lstm():
    start_date = '2015-02-11'
    end_date = '2015-03-01'
    df = get_real_pericted(start_date, end_date)
    dict_data = df.to_dict(orient='records')
    return render_template('lstm.html', data=dict_data)

@app.route("/neuro/rnn")
def rnn():
    predict = predict_spam()
    return render_template('RNN.html', predict=predict)

@app.route("/neuro/gradient")
def gradient():
    random_data_processed, y_random_pred = grad_random()
    return render_template('gradient.html', real=random_data_processed, pred =y_random_pred)

@app.route("/neuro/lstm/study")
def lstm_study():
    flash(study_lstm(1))
    return redirect("/neuro/lstm")

@app.route("/neuro/rnn/study")
def rnn_study():
    flash(study_rnn(1))
    return redirect("/neuro/rnn")

@app.route("/neuro/gradient/study")
def gradient_study():
    flash(study_grad())
    return redirect("/neuro/gradient")