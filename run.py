#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app import app, db
from flask import make_response, jsonify
from app.ocr.ocr import recreate_dataset
from app.ann import create_ann_dataset

@app.route('/about')
def about():
    return 'About Us'


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.before_first_request
def servermake():
    create_ann_dataset()
    recreate_dataset()

def main():
    db.create_all()
    app.run(debug=True)


if __name__ == '__main__':
    main()