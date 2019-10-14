from app import db

class TextArea (db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    minx = db.Column(db.Integer, nullable=False)
    miny = db.Column(db.Integer, nullable=False)
    maxx = db.Column(db.Integer, nullable=False)
    maxy = db.Column(db.Integer, nullable=False)
    jtext = db.Column(db.String(50), nullable=True)
    romaji = db.Column(db.String(50), nullable=True)
    etext = db.Column(db.String(50), nullable=True)
    roidata = db.Column(db.LargeBinary)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'))