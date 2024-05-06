from flask import Blueprint

extraction_blueprint = Blueprint('extraction', __name__)


@extraction_blueprint.route('/helloWorld', methods=['GET'])
def hello_world():
    return 'Hello, World!'
