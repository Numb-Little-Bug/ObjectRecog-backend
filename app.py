"""
程序的入口，可以在这里启动服务
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from api import api_bp
import config

app = Flask(__name__)
app.config.from_object(config)
db = SQLAlchemy(app)

app.register_blueprint(api_bp, url_prefix='/api')


if __name__ == '__main__':
    app.run(debug=True)
