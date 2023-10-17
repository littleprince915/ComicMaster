#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import redirect, render_template, request, make_response
from flask_login import login_required, current_user
from app.database.imagedata import gettextareadata, getimagefiledata, createimagedata, getimagedata
from app import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/imagedata", methods=["GET"])
@login_required
def uploadimgdataget():
    return render_template("imagedata.html", withimage=False)


@app.route("/imagedata", methods=["POST"])
@login_required
def uploadimgdatapost():
    file = request.files['file']

    if file.filename == '':
        return render_template("imagedata.html", withimage=False)

    if file and allowed_file(file.filename):
        newimageid = createimagedata(file.read())

        return redirect("/imagedata/{}".format(newimageid))

    return render_template("imagedata.html", withimage=False)


@app.route("/imagedata/<int:imageid>", methods=["GET"])
@login_required
def getimagedataboxes(imageid):
    imagedata = getimagedata(imageid)

    return render_template("imagedata.html", boxes=imagedata, imageid=imageid, withimage=True)


@app.route("/imagefiledata/<int:imageid>", methods=["GET"])
@login_required
def imagedatafile(imageid):
    db_image = getimagefiledata(imageid)

    response = make_response(db_image) #this function accepts binary image
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename='%s.jpg' % imageid)

    return response


@app.route("/textareadata/<int:textareaid>", methods=["GET"])
@login_required
def textareadatafile(textareaid):
    db_textarea = gettextareadata(textareaid)

    response = make_response(db_textarea) #this function accepts binary image
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename='%s.jpg' % textareaid)

    return response