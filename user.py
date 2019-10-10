#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import redirect, render_template, request, Blueprint, session, url_for
from flask_login import login_user, login_required, logout_user, current_user
from datetime import timedelta
from database.user import User, createuser
from . import login_manager

users = Blueprint('users', __name__)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    redirect("/user/login")


@users.route('/home', methods=["GET"])
@login_required
def home():
    render_template("home.html")


@users.route('/user/login', methods=["GET"])
def userloginget():
    if current_user.is_authenticated():
        return redirect('/home')

    return render_template('login.html')


@users.route('/user/login', methods=["POST"])
def userloginpost():
    uname = request.form['nm']
    password = request.form['pw']

    try:
        user = User.query.filter_by(username=uname).first()
        password = password.strip()
        password2 = user.password.strip()

        if password == password2:
            login_user(user, remember=True, duration=timedelta(days=5))
            return redirect("/home")

    except:
        return render_template("login.html")


@users.route('/user/register', methods=["GET"])
def userregisterget():
    return render_template('register.html')


@users.route('/user/register', methods=["POST"])
def userregisterpost():
    if current_user.is_authenticated():
        logout_user()

    username = request.form['nm']
    password = request.form['pw']
    email = request.form['eadd']

    new_user = createuser(username, password, email)
    login_user(new_user, remember=True, duration=timedelta(days=5))
    return redirect("/home")


@users.route('/user/logout', methods=["POST"])
def userlogout():
    if current_user.is_authenticated():
        logout_user()

    return render_template('login.html')