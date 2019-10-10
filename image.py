#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import redirect, render_template, request, Blueprint, session, url_for, make_response
from flask_login import login_user, login_required, logout_user, current_user
from database.image import Image, createimage, getuserimages, getimage, getimagefile
from . import login_manager

image = Blueprint('image', __name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@image.route("/image", methods=["GET"])
@login_required
def image():
    userid = current_user.get_id()
    images = getuserimages(userid)

    return render_template("gallery.html", images=images)


@image.route("/image/{imageid}", methods=["GET"])
@login_required
def imageid(imgid):
    imagedata = getimage(imgid)

    return render_template("index.html", boxes=imagedata)


@image.route("/imagefile/{imageid}", methods=["GET"])
@login_required
def imagefile(imgid):
    db_image = getimagefile(imgid)

    response = make_response(db_image) #this function accepts binary image
    response.headers.set('Content-Type', 'image/jpeg') #should be accepting jpg, jpeg, png, gif
    response.headers.set('Content-Disposition', 'attachment', filename='%s.jpg' % imgid)

    return response


@image.route("/image", methods=["POST"])
@login_required
def uploadimg():
    file = request.files['file']

    if file.filename == '':
        return render_template("index.html")

    if file and allowed_file(file.filename):
        userid = current_user.get_id()
        newimageid = createimage(file.filename, file.read, userid)

        return redirect("/image/{}".format(newimageid))

    return render_template("index.html")