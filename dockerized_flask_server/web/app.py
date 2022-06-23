from flask import Flask
from flask_restful import Api
from flask_cors import CORS

# local imports
from utils import (
    Register, Fragmentation
)

# Initalize Flask & Flask Restful Objects
app = Flask(__name__)
api = Api(app)

# Enabling CORS
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Mapping resources/classes to relevant endpoints 
api.add_resource(Register, "/register")
api.add_resource(Fragmentation, "/fragment")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)