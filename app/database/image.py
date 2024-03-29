from sqlalchemy.orm import load_only
from app.database.user import getuserbyid
from app.database.textarea import TextArea
from app.textdetect import get_text_areas, get_text_areas_only, ann_classify, get_characters
from app import db

import cv2

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    data = db.Column(db.LargeBinary)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.acctNo'))
    textareas = db.relationship('TextArea', backref='image')


def createimage(data, userid):
    user = getuserbyid(userid)
    resized_image, textareas = get_text_areas(data)
    data = cv2.imencode('.jpg', resized_image)[1].tostring()

    newImage = Image(data=data, owner=user)
    db.session.add(newImage)
    db.session.commit()

    for textarea in textareas:
        newtextarea = TextArea(minx=textarea["minx"], miny=textarea["miny"], maxx=textarea["maxx"], maxy=textarea["maxy"],
                               jtext=textarea["jtext"], romaji=textarea["romaji"], etext=textarea["etext"], image=newImage,
                               roidata=textarea["roidata"], isjapanese=True)
        db.session.add(newtextarea)
    db.session.commit()

    return newImage.id


def getuserimageids(userid):
    imageids = Image.query.filter_by(owner_id=userid).all()
    imageids = [x.id for x in imageids]

    return imageids


def getimage(imageid):
    image = Image.query.get(int(imageid))

    return image.textareas


def getimagefile(imageid):
    image = Image.query.get(int(imageid))

    return image.data


def gettextarea(textareaid):
    textareadata = TextArea.query.get(int(textareaid))

    return textareadata.roidata


def createimagestep1(data, userid):
    user = getuserbyid(userid)
    resized_image, textareas = get_text_areas_only(data)
    data = cv2.imencode('.jpg', resized_image)[1].tostring()

    newImage = Image(data=data, owner=user)
    db.session.add(newImage)
    db.session.commit()

    for textarea in textareas:
        newtextarea = TextArea(minx=textarea["minx"], miny=textarea["miny"], maxx=textarea["maxx"], maxy=textarea["maxy"],
                               jtext=textarea["jtext"], romaji=textarea["romaji"], etext=textarea["etext"], image=newImage,
                               roidata=textarea["roidata"])
        db.session.add(newtextarea)
    db.session.commit()

    return newImage.id


def createimagestep2(imageid):
    image = Image.query.get(int(imageid))
    textareas = image.textareas

    for textarea in textareas:
        roibytes = textarea.roidata
        isjapanese = ann_classify(roibytes)

        textarea.isjapanese = isjapanese
        db.session.add(textarea)
    db.session.commit()

    return image.id


def createimagestep3(imageid):
    image = Image.query.get(int(imageid))
    textareas = image.textareas

    for textarea in textareas:
        roibytes = textarea.roidata

        if not textarea.isjapanese:
            continue

        jtext, etext, romaji = get_characters(roibytes)

        if not jtext:
            continue

        textarea.jtext = jtext
        textarea.etext = etext
        textarea.romaji = romaji
        db.session.add(textarea)

    db.session.commit()

    return image.id