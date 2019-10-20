#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import redirect, render_template, request, make_response
from flask_login import login_required, current_user
from app.database.image import createimage, getuserimageids, getimage, createimagestep1, createimagestep2, createimagestep3
from app import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/imagestep", methods=["GET"])
@login_required
def uploadimgstepget():
    return render_template("imagestep.html", withimage=False)


@app.route("/imagestep", methods=["POST"])
@login_required
def uploadimgsteppost():
    file = request.files['file']

    if file.filename == '':
        return render_template("imagestep.html", withimage=False)

    if file and allowed_file(file.filename):
        userid = current_user.get_id()
        newimageid = createimagestep1(file.read(), userid)

        return redirect("/imagestep/{}/step1".format(newimageid))

    return render_template("imagestep.html", withimage=False)


@app.route("/imagestep/<int:imageid>/step1", methods=["GET"])
@login_required
def getimagestep1(imageid):
    imagedata = getimage(imageid)
    print imagedata

    return render_template("imagestep1.html", boxes=imagedata, imageid=imageid)


@app.route("/imagestep/<int:imageid>/step2", methods=["GET"])
@login_required
def getimagestep2(imageid):
    imagedata = getimage(imageid)

    return render_template("imagestep2.html", boxes=imagedata, imageid=imageid, withimage=True)


@app.route("/imagestep/<int:imageid>/step2", methods=["POST"])
@login_required
def postimagestep2(imageid):
    createimagestep2(imageid)

    return redirect("/imagestep/{}/step2".format(imageid))


@app.route("/imagestep/<int:imageid>/step3", methods=["GET"])
@login_required
def getimagestep3(imageid):
    imagedata = getimage(imageid)

    return render_template("imagestep3.html", boxes=imagedata, imageid=imageid, withimage=True)


@app.route("/imagestep/<int:imageid>/step3", methods=["POST"])
@login_required
def postimagestep3(imageid):
    createimagestep3(imageid)

    return redirect("/imagestep/{}/step3".format(imageid))