FROM yoanlin/opencv-python3
ADD . /code
WORKDIR /code
RUN apt-get install -y libjpeg62-turbo-dev
COPY ./data /code/data
RUN pip install -r requirements.txt
CMD python app.py