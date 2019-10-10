#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask_login import LoginManager
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
database_file = "sqlite:///{}".format(os.path.join(project_dir, 'users_database.db'))
upload_dir = os.path.join(project_dir, "./static/")

app = Flask(__name__)
app.secret_key = os.urandom(50)
app.config['SQLALCHEMY_DATABASE_URI'] = database_file
app.config['UPLOAD_FOLDER'] = upload_dir

login_manager = LoginManager()
login_manager.init_app(app)