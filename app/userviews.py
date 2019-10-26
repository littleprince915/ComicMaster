#!/usr/bin/env python
# -*- coding: utf-8 -*

from app import app
from flask import redirect, render_template, request, Blueprint
from flask_login import login_user, login_required, logout_user, current_user
from datetime import timedelta
from app.database.user import createuser, getuser, getuserbyid
from app import login_manager

users = Blueprint('users', __name__)


@login_manager.user_loader
def load_user(user_id):
    return getuserbyid(user_id)


@login_manager.unauthorized_handler
def unauthorized():
    return redirect("/user/login")


@app.route('/home', methods=["GET"])
@app.route('/', methods=["GET"])
@login_required
def home():
    return render_template("index.html", withimage=False)


@app.route('/user/login', methods=["GET"])
def userloginget():
    if current_user.is_authenticated:
        return redirect('/home')

    return render_template('login.html')


@app.route('/user/login', methods=["POST"])
def userloginpost():
    uname = request.form['username']
    password = request.form['password']

    try:
        user = getuser(uname, password)
        password = password.strip()
        password2 = user.password.strip()

        if password == password2:
            login_user(user, remember=True, duration=timedelta(days=5))
            return redirect("/home")

    except:
        return render_template("login.html")


@app.route('/user/register', methods=["GET"])
def userregisterget():
    return render_template('register.html')


@app.route('/user/register', methods=["POST"])
def userregisterpost():
    if current_user.is_authenticated:
        logout_user()

    username = request.form['username']
    password = request.form['password']
    email = request.form['emailadd']

    new_user = createuser(username, password, email)
    login_user(new_user, remember=True, duration=timedelta(days=5))
    return redirect("/home")


@app.route('/user/logout', methods=["GET"])
def userlogout():
    if current_user.is_authenticated:
        logout_user()

    return render_template('login.html')
