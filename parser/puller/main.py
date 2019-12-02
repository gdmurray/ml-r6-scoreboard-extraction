import lxml
from lxml import etree
from urllib import request
import cv2
import streamlink
import os
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import concurrent.futures


def process_video(url):
    youtube = etree.HTML(request.urlopen(url).read())
    video_title = youtube.xpath("//span[@id='eow-title']/@title")
    video_title = ''.join(video_title)
    video_id = url.split("=")[-1]

    folder_name = "".join([c for c in video_title if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
    folder_name = "_".join([c for c in folder_name.split(" ")])

    # folder_name = "_".join([c for c in folder_name.split(" ")])
    print(f"Processing Video {video_id} with title: {video_title}")
    print(f"In Folder: {folder_name}")
    directory = "/home/ubuntu/data/{}".format(folder_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        stream = streamlink.streams(url)
        s = stream['best']

        vcap = cv2.VideoCapture(s.url)
        count = 0

        sec = 0
        frameRate = 2

        def getFrame(sec):
            vcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vcap.read()
            if hasFrames:
                cv2.imwrite(f"{directory}/frame_{video_id}_{count}.jpg", image)
            else:
                print("Does not have frames")
            return hasFrames

        p_hundreds = 0
        success = getFrame(sec)
        while success:
            sec = sec + frameRate
            sec = round(sec, 2)
            count += 1
            success = getFrame(sec)

            hundreds = round(count / 100, 0)
            if hundreds > p_hundreds:
                p_hundreds = hundreds
                print("{}: Created {} frames".format(video_title, hundreds * 100))

    except Exception as e:
        traceback.print_exc()
        print(f"ERROR IN {video_title}: {e}")
        return f"ERROR: {e}"

    return "SUCCESS"


with open("NA_games.txt", 'r') as f:
    videos = [t.replace("\n", '').strip() for t in f.readlines()]

print(videos)
print(f"Videos length: {len(videos)}")
start = time.time()

with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_url = {executor.submit(process_video, url): url for url in videos}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            status = future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")
        else:
            print(f"{url} Completed: {status}")

finish = time.time()
print(f"Finished Processing {len(videos)} Videos in: {finish - start:.2f}")
