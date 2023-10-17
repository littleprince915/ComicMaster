from app import db

class TextAreaData (db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    minx = db.Column(db.Integer, nullable=False)
    miny = db.Column(db.Integer, nullable=False)
    maxx = db.Column(db.Integer, nullable=False)
    maxy = db.Column(db.Integer, nullable=False)
    isjapanese = db.Column(db.Boolean, default=False)
    roidata = db.Column(db.LargeBinary)
    imagedata_id = db.Column(db.Integer, db.ForeignKey('image_data.id'))