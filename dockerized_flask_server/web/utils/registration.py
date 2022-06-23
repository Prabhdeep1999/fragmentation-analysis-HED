from flask_restful import Resource
from flask import jsonify, request
import bcrypt

# local imports
from .helper_utils import check_input, user_exists, update
from .postgres_connection import cursor, connection

class Register(Resource):
    """Performs Registration of a user

    Args:
        Resource (flask_restful.Resource): To perform POST API in a RESTful way
    """
    def post(self):
        # get data from user
        postedData = request.get_json()

        # get status code
        ret = check_input(postedData)
        if ret["status"] == 200:
                pass
        else:
            return jsonify(ret)

        username = str(postedData["username"])
        password = str(postedData["password"])

        # check if user exists
        ret = user_exists((username, ), cursor)
        if ret == "No Problem":
           pass
        else:
           return jsonify(ret)

        # hashing the password
        hashed_pw = bcrypt.hashpw(password.encode("utf8"), bcrypt.gensalt()).decode('utf8')

        # inserting user to database
        update((username, hashed_pw), cursor, connection)

        # returning response
        retJson = {
            "status": 200,
            "msg": "You successfully registered!"
        }

        return jsonify(retJson)