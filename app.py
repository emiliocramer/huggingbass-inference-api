from flask import Flask
from routes.extraction import extraction_blueprint
from routes.inference import inference_blueprint
from routes.comparison import comparison_blueprint

app = Flask(__name__)

# Register blueprints
app.register_blueprint(extraction_blueprint)
app.register_blueprint(inference_blueprint)
app.register_blueprint(comparison_blueprint)

if __name__ == '__main__':
    app.run()