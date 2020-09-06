#!pip install decord

import numpy as np
import pandas as pd
import cv2
import os
from google.colab.patches import cv2_imshow
import imageio
from decord import VideoReader
import shutil
from decord import cpu, gpu
import moviepy
import moviepy.editor
import shutil

imageio.plugins.ffmpeg.download()
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

FPATH = "./"
markup = pd.read_csv(f'{FPATH}/trailers.csv')
movie_list = markup['Name'].values

"""**Making sublcips (30 seconds duration)**"""
#Check if a subclip was correctly created, otherwise delete it
for film_dir in os.listdir(f"{FPATH}"):
    #Check whether film_dir is directory, but not film trailer
    if os.path.isdir(f"{FPATH}/{film_dir}") and film_dir !='TrainData':
        for subclip in os.listdir(f"{FPATH}/{film_dir}"):
            try:
                video = moviepy.editor.VideoFileClip(f"{FPATH}/{film_dir}/{subclip}")
                if int(video.duration) <= 20:
                  raise DurationError
            except:
                print(f"The subclip {subclip} of {film_dir} was not correctly created, deleting...")
                os.remove(f"{FPATH}/{film_dir}/{subclip}")

"""**Making separate frames from each subcip**"""

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    # load the VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)
                     
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0
    
    frames = vr.get_batch(frames_list).asnumpy()

    for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
        save_path = os.path.join(frames_dir, video_filename.split('.')[0], "{:010d}.jpg".format(index))  # create the save path
        if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # save the extracted image
            saved_count += 1  # increment our counter by one

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=1):
    """
    Extracts the frames from a video
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename.split('.')[0]), exist_ok=True)
    
    print("Extracting frames from {}".format(video_filename))
    
    extract_frames(video_path, frames_dir, every=every)  # let's now extract the frames

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames

def preprocess(path, train_path, movie_name, n_subclips=3, subclip_duration=30, frequency=45, verbose=False):
  
    """
      Preprocesses a movie trailer making subclips and then extracting sequences of frames
      :param path: path to the movie trailer
      :param train_path: path to the directory with sequences
      :param movie_name: movie name
      :param n_subclips: number of subclips to make from the trailer
      :param subclip_duration: duration of a subclip
      :param frequency: frequency of extracting frames from subclips
      :param verbose: increase verbosity if True
    """
    name = '.'.join(movie_name.split('.')[:-1])
    format = movie_name.split('.')[-1]
    if format == 'flv':   #Decord does not work with flv format
        format = 'mov'

    #Extracting subclip from trailer
    base = 10

    os.mkdir(f"{path}/{name}")
    for i in range(n_subclips): 
        if verbose:
            print(f"{i} iteration...")
            print("....Making subclip....")
        try:
            ffmpeg_extract_subclip(f"{path}/{movie_name}", base, base+subclip_duration, targetname=f"{path}/{name}/{i}.{format}")
            base = base + subclip_duration
        except BaseException:
            print(f"Some error occured during {i+1} extraction")
            continue

        #Check if all subclips were correctly created
        try:
            video = moviepy.editor.VideoFileClip(f"{path}/{name}/{i}.{format}")
            if int(video.duration) <= subclip_duration//2:
                raise DurationError
        except:
            print(f"The {i} subclip was not correctly created, deleting...")
            os.remove(f"{path}/{name}/{i}.{format}")
            continue

        #Creating frames
        if verbose:
            print("....Extracting frames....")
        os.makedirs(f"{train_path}/{name+'_'+str(i)}", exist_ok=True)   #Creating directory for Train dataset
        try:
            video_to_frames(f"{path}/{name}/{i}.{format}", f"{train_path}/{name+'_'+str(i)}", overwrite=False, every=frequency)
        except:
            print("Error occured while executing VIDEO_TO_FRAMES")
            os.rmdir(f"{train_path}/{name+'_'+str(i)}/{i}")
            os.rmdir(f"{train_path}/{name+'_'+str(i)}")
            continue

    #Delete directory with subclips
    if name in os.listdir(f"{path}"):   
        shutil.rmtree(f"{path}/{name}")

def movie_preprocessing(path, train_path, movie_list, n_subclips=3, subclip_duration=30, frequency=45, verbose=False):
    for movie_name in movie_list:
        print(f"Preprocessing {movie_name}")
        preprocess(path, train_path, movie_name, n_subclips, subclip_duration, frequency, verbose)
        

if __name__ == '__main__':
    movie_preprocessing(f'{FPATH}/100trailers', f'{FPATH}/TrainData', movie_list=movie_list, verbose=True)
