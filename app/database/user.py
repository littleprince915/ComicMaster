from flask_login import UserMixin
from app import db


class User(UserMixin, db.Model):
    acctNo = db.Column(db.Integer, primary_key=True, nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=True)
    password = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    images = db.relationship('Image', backref='owner')

    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email

    def __repr__(self):
        return 'User {}'.format(self.username)

    def get_id(self):
        return (self.acctNo)


def createuser(username, password, email):
    new_user = User(username=username, password=password, email=email)
    db.session.add(new_user)
    db.session.commit()

    return new_user


def getuser(username, password):
    user = User.query.filter_by(username=username).first()

    return user


def getuserbyid(user_id):
    user = User.query.get(int(user_id))

    return user
