import boto3
from botocore.exceptions import ClientError
import bcrypt


def check_input(postedData):
    """Sanitary Check for the data received from user

    Args:
        postedData (dict): Data from user in a dictionary format

    Returns:
        retJSon (dict): A dictionary with relevant status code and message is returned
    """
    # check everything is okay
    if postedData["username"] and postedData["password"]:
        retJson = {
            "status": 200,
            "msg": "Okay"
        }
        return retJson
    
    # check for both id and pass present
    else:
        retJson = {
            "status": 304,
            "msg": "Please input both ID and pass"
        }
        return retJson


def user_exists(user_name, cursor):
    """Check if user exists in the database

    Args:
        user_name (tuple): username
        cursor (_cursor): Cursor object of Postgresql database

    Returns:
        retJSon (dict): A dictionary with relevant status code and message is returned
    """

    query = "select user_name from auth where user_name = %s"
    try:
        cursor.execute(query, user_name)

        # fetch results
        ret = cursor.fetchone()[0]

        print(ret, cursor)

        if ret != None:
            retJson = {
                "status": 301,
                "msg": "User already exists"
            }
            return retJson

    except Exception as e:
        print(e, 2)
        return "No Problem"


def verify(cursor, uname=None, passw=None):
    """_summary_

    Args:
        cursor (_cursor): Cursor object of Postgresql database
        uname (str, optional): Username. Defaults to None.
        passw (str, optional): Password. Defaults to None.

    Returns:
        retJSon (dict): A dictionary with relevant status code and message is returned
    """

    if uname == None or passw == None:
        retJson = {
            "status": 304,
            "msg": "Please input ID and pass"
        }
        return retJson

    # check for valid uname and pass:
    query = "select password from auth where user_name = %s"
    try:
        cursor.execute(query, (uname,))
        hashed_pw = cursor.fetchone()[0]
        passw = passw.encode('utf-8')
        hashed_pw = bytes(hashed_pw, 'utf-8')
        if hashed_pw != None:
            if bcrypt.hashpw(passw, hashed_pw) == hashed_pw:
                retJson = {
                    "status": 200,
                    "msg": "ID and pass matches"
                }
                return retJson
            else:
                retJson = {
                    "status": 302,
                    "msg": "Incorrect User ID or password"
                }
                return retJson
    except Exception as e:
        print(e)
        retJson = {
            "status": 302,
            "msg": "Incorrect User ID or Password"
        }
        return retJson

def update(val, cursor, connection):
    """Update a value in the database

    Args:
        val (tuple): value to be updated
        cursor (_cursor): Cursor object of Postgresql database
        conn (connection): Connetion object of Postgresql database
    """

    query = "insert into auth values(%s,%s);"
    # insert values into table
    cursor.execute(query, val)

    # commit the changes to database
    connection.commit()

def upload_file_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    Args:
        file_name (str): File to upload
        bucket (str): Bucket to upload to
        object_name (str, optional): S3 object name. If not specified then file_name is used. Defaults to None.

    Returns:
        result (bool): True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name
    
    # Let's use Amazon S3
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True