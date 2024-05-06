from flask import Blueprint
from db import songs_collection

extraction_blueprint = Blueprint('extraction', __name__)


@extraction_blueprint.route('/helloWorld', methods=['GET'])
def hello_world():
    first_song = songs_collection.find_one()
    return str(first_song)
