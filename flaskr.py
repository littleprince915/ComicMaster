#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os

from flask import redirect, render_template, request, make_response, jsonify

import ocr
from . import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def loginpage():
    return render_template('login.html')
    
    
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect("/index")
    file = request.files['file']
    
    if file.filename == '':
        return redirect("/index")
        
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], "original_manga.jpg"))
        return redirect("/index/12")
        
    else:
        return redirect(request.url)
    

@app.route('/login')
def login():
    #user = User.query.filter_by(username='Anthony').first()
    #login_user(user)
    return redirect("/index")
    
    
@app.route('/logout')
def logout():
    # logout_user()
    return 'You are now logged out!'
    
    
@app.route('/gallery')
def gallery():

    return render_template('gallery.html')

    
@app.route('/index', defaults={'imageid': None})
@app.route('/index/<imageid>')
def index(imageid):
    if (imageid != None):
    
        boxes = ocr.detect_boxes()
        
        return render_template('index.html', withimage=True, imageid=imageid, boxes=boxes)

    return render_template('index.html', withimage=False)

    
# @app.route('/image/<imageid>')
# def image(imageid):
#     #file_data = Image.query.filter_by(id=imageid).first()
#
#     return send_file(BytesIO(file_data.imagefile), attachment_filename='{}.{}'.format(file_data.id, file_data.extension))
    
    
@app.route('/register')
def register():
    return render_template('register.html')

    
@app.route('/training')
def training():
    retraining = False

    if(not retraining):
        return render_template('training.html', retraining=False)

    return render_template('training.html', retraining=True)

    
@app.route('/about')
def about():
    return 'About Us'
    
    
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
