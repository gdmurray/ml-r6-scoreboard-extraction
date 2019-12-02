import numpy as np
import argparse
import imutils
import itertools
import cv2

MIN_HEIGHT = 10


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])
