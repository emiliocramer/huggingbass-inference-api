from flask import Flask
from flask_cors import CORS
from routes.extraction import extraction_blueprint
from routes.inference import inference_blueprint
from routes.comparison import comparison_blueprint
from routes.remix import remix_blueprint
from routes.juice_vrse import juice_blueprint

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Register blueprints
app.register_blueprint(extraction_blueprint)
app.register_blueprint(inference_blueprint)
app.register_blueprint(comparison_blueprint)
app.register_blueprint(remix_blueprint)
app.register_blueprint(juice_blueprint)


if __name__ == '__main__':
    app.run()