#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os

from flask import redirect, render_template, request, make_response, jsonify
from . import app

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