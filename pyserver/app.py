from flask import Flask
from redis import Redis
import numpy as np
import cv2
import os
import requests
import random
import base64
import math
import time
import uuid

app = Flask(__name__)
r = Redis(host='redis', port=6379)
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 50


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
