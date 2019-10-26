#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import redirect, render_template, request, make_response
from flask_login import login_required, current_user
from app.database.image import createimage, getuserimageids, getimage, getimagefile, gettextarea
from app import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/gallery", methods=["GET"])
@login_required
def imagesofuser():
    userid = current_user.get_id()
    imageids = getuserimageids(userid)

    return render_template("gallery.html", imageids=imageids)


@app.route("/image", methods=["GET"])
@login_required
def uploadimgget():
    return render_template("index.html", withimage=False)


@app.route("/image", methods=["POST"])
@login_required
def uploadimgpost():
    file = request.files['file']

    if file.filename == '':
        return render_template("index.html", withimage=False)

    if file and allowed_file(file.filename):
        userid = current_user.get_id()
        newimageid = createimage(file.read(), userid)

        return redirect("/image/{}".format(newimageid))

    return render_template("index.html", withimage=False)


@app.route("/image/<int:imageid>", methods=["GET"])
@login_required
def getimageboxes(imageid):
    imagedata = getimage(imageid)

    return render_template("index.html", boxes=imagedata, imageid=imageid, withimage=True)


@app.route("/imagefile/<int:imageid>", methods=["GET"])
@login_required
def imagefile(imageid):
    db_image = getimagefile(imageid)

    response = make_response(db_image)  # this function accepts binary image
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename='%s.jpg' % imageid)

    return response


@app.route("/textarea/<int:textareaid>", methods=["GET"])
@login_required
def textareafile(textareaid):
    db_textarea = gettextarea(textareaid)

    response = make_response(db_textarea)  # this function accepts binary image
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename='%s.jpg' % textareaid)

    return response
