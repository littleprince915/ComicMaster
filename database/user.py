from flask_login import UserMixin
from . import db


class User (UserMixin, db.Model):
    acctNo = db.Column(db.Integer, primary_key=True, nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=True)
    password = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    imagefile = db.relationship('ImageFiles', backref='owner')

    def __init__(self, username, password, email):

        self.username = username
        self.password = password
        self.email = email

    def __repr__(self):
        return 'User <>'.format(self.username)


def createuser(username, password, email):
    new_user = User(username=username, password=password, email=email)
    db.session.add(new_user)
    db.session.commit()
    #db.session.close()

    return new_user
