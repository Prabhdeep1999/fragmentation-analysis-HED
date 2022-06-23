from flask_restful import Resource
from flask import jsonify, request
from werkzeug.utils import secure_filename
import os
import time
from random import choices
from string import ascii_uppercase, digits
import cv2
from skimage.measure import block_reduce
import sys
import numpy as np

# local imports
from .helper_utils import check_input, verify
from .fragmentation_helper import base64_to_image, processing, image_to_base64, OUTPUT_FRAG_RESULTS_CV_CSV, OUTPUT_FRAG_IMG
from .postgres_connection import cursor


class Fragmentation(Resource):
    """Performs the fragmentation analysis on base64 image and returns output base64 image along with other useful data.

    Args:
        Resource (flask_restful.Resource): To perform POST API in a RESTful way
    """
    def post(self):

        # fetching data from FORM type data
        postedData = request.get_json()
        username = postedData["username"]
        password = postedData["password"]

        # check validity
        ret1 = check_input({"username": username, "password": password})

        # verify credentials
        ret2 = verify(cursor, username, password)
        if ret1["status"] != 200:
            return jsonify(ret1)
        else:
            if ret2["status"] != 200:
                return jsonify(ret2) 

        # starting time to record time required for operation
        start = time.time()

        # storing base64 string from JSON
        image_base64 = postedData["image"]

        # converting base64 string to image
        image = base64_to_image(image_base64)

        # generating random string for filename using choices module from random library 
        # and ascii_uppercase & digits from string library
        filename_ = ''.join(choices(ascii_uppercase + digits, k = 10))
        filename = filename_ + ".jpg"
        
        # storing image for further operation
        image.save(secure_filename(filename))

        # Converting large images to small
        size_in_mb = os.path.getsize(filename) * (10 ** -6)

        # For optimization purpose if image is of seize greater than 0.9 MB refuce the size
        if size_in_mb > 0.9:
            img = cv2.imread(filename)
            print("Size of image before block reduce: ", img.shape, file=sys.stderr)
            image_small = block_reduce(img, block_size=(2,2,1), func=np.mean)
            print("Size of image after block reduce: ", image_small.shape, file=sys.stderr)
            cv2.imwrite(filename, image_small)

        # storing Low Intesity and High Intensity from Image, These are the thresholding intensities
        low_intensity = int(postedData["LI"])
        high_intensity = int(postedData["HI"])

        # Processing image using HED algo
        ret = processing(filename, low_intensity, high_intensity)

        # Preparing and returning Response
        retJson = {
            "status": 200,
            "msg": image_to_base64(ret[0]),
            "bandw": image_to_base64(ret[2]),
            "data": ret[1],
            "time consumed": time.time() - start
        }

        # Remove the temp files generated
        os.remove(filename)
        os.remove(filename[:9] + OUTPUT_FRAG_RESULTS_CV_CSV)
        os.remove(filename[:9] + OUTPUT_FRAG_IMG)
        os.remove(ret[2])

        return jsonify(retJson)
