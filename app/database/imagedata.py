from sqlalchemy.orm import load_only
from app.database.user import getuserbyid
from app.database.textareadata import TextAreaData
from app.textdetect import get_text_areas_without_ann
from app import db

import cv2

class ImageData(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    data = db.Column(db.LargeBinary)
    textareas = db.relationship('TextAreaData', backref='imagedata')


def createimagedata(data):

    resized_image, textareas = get_text_areas_without_ann(data)

    data = cv2.imencode('.jpg', resized_image)[1].tostring()

    newImage = ImageData(data=data)
    db.session.add(newImage)
    db.session.commit()

    for textarea in textareas:
        newtextarea = TextAreaData(minx=textarea["minx"], miny=textarea["miny"], maxx=textarea["maxx"], maxy=textarea["maxy"],
                                   imagedata=newImage, roidata=textarea["roidata"], isjapanese=False)
        db.session.add(newtextarea)
    db.session.commit()

    return newImage.id


def getimagedata(imageid):
    image = ImageData.query.get(int(imageid))

    return image.textareas


def getimagefiledata(imageid):
    image = ImageData.query.get(int(imageid))

    return image.data


def gettextareadata(textareaid):
    textareadata = TextAreaData.query.get(int(textareaid))

    return textareadata.roidata