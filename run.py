from app import app, db
from flask import make_response, jsonify


@app.route('/about')
def about():
    return 'About Us'


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    db.create_all()
    app.run(debug=True)


if __name__ == '__main__':
    main()