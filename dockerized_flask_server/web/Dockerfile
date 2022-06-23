FROM jjanzic/docker-python3-opencv
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
# RUN install_docker.sh
COPY . .
CMD ["python3", "app.py"]