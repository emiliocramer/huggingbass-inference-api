from flask import Flask
from routes.extraction import extraction_blueprint

app = Flask(__name__)

# Register blueprints
app.register_blueprint(extraction_blueprint)

if __name__ == '__main__':
    app.run(debug=True, port=3002)