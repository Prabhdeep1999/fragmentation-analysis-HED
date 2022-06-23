# fragmentation-analysis-HED

Fragmentation Analysis is a key check used by mining engineers after blasting to determine the efficacy of blast or blast accuracy. It focuses on checking the average size of rocks/fragments generated after blast, This is a image processing / computer vision approach using Holistically Nested Edge Detection Algorithm (HED)

This is a standard usecase and can be used to find size of unknown objects with clear boundaries in terms of pixels in a stabilised camera with consistent camera position.

Following is a series of snap showing the three phases (Raw | Fragmented Image)
`<img src="image/README/demo_image.jpg" alt="drawing" width="480" height="270"/><img src="image/README/out.jpg" alt="drawing" width="480" height="270"/>`

1. Python3.7
2. If Docker Approach is selcted:
   1. Docker
   2. Docker Compose

The program can run in either of the two following ways:

1. Running the python file directly by passing the video url with the python file (Put your images in the images folder to save all output in the same folder):

   ```
   pip install -r dockerized_flask_server/web/requirements.txt
   ```

   ```
   python fragmentation_analysis.py --input ./images/Readme/demo_image.jpg
   ```
2. I've also made it as a complete dockerized server with proper authentication.
   Following is the way to run the dockerized version:

   ```
   cd dockerized_flask_container
   ```

   ```
   sudo docker-compose build
   ```

   ```
   sudo docker-compose up
   ```

*Resource Chart for Rest API:*

| Resources | URL       | Method | Param                          | Status                                                                     | Param Body type |
| --------- | --------- | ------ | ------------------------------ | -------------------------------------------------------------------------- | --------------- |
| Register  | /register | post   | uname, pass                    | 200 OK, 301 username already exist                                         | JSON            |
| Classify  | /fragment | post   | uname, pass, image_file_base64 | 200 OK, 301, 302 Incorrect id or pass, 303 Out of token, 304 Invalid input | Form            |

*Database.ini:*

I have used a public database but you can replace it with your credentials in the **web/req_files/database.ini** file
