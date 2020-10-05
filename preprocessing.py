#!pip install decord

import numpy as np
import pandas as pd
import cv2
import os
import imageio
from decord import VideoReader
import shutil
from decord import cpu, gpu
import moviepy
import moviepy.editor
import shutil
from config import FPATH, TRAINING_PATH, PREDICTION_PATH, MARKUP_PATH

# imageio.plugins.ffmpeg.download()
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


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

def preprocess(movie_name, train=True, n_subclips=3, subclip_duration=30, frequency=45, verbose=False):
  
    """
      Preprocesses a movie trailer making subclips and then extracting sequences of frames
      :param movie_name: movie name
      :train: boolean flag to determine whether preprocessing is performed for training videos or not
      :param n_subclips: number of subclips to make from the trailer
      :param subclip_duration: duration of a subclip
      :param frequency: frequency of extracting frames from subclips
      :param verbose: increase verbosity if True
    """
    if train == True:
        if not os.path.isdir(TRAINING_PATH):
            os.mkdir(TRAINING_PATH)
        DEST = TRAINING_PATH
    else:
        if not os.path.isdir(PREDICTION_PATH):
            os.mkdir(PREDICTION_PATH)
        DEST = PREDICTION_PATH
        
    name = '.'.join(movie_name.split('.')[:-1])
    format = movie_name.split('.')[-1]
    if format == 'flv':   #Decord does not work with flv format
        format = 'mov'

    #Extracting subclip from trailer
    base = 10

    os.makedirs(f"{FPATH}/{name}", exist_ok=True)
    for i in range(n_subclips): 
        if verbose:
            print(f"{i} iteration...")
            print("....Making subclip....")
        try:
            ffmpeg_extract_subclip(f"{FPATH}/{movie_name}", base, base+subclip_duration, targetname=f"{FPATH}/{name}/{i}.{format}")
            base = base + subclip_duration
        except BaseException:
            print(f"Some error occured during {i+1} extraction")
            continue

        #Check if all subclips were correctly created
        try:
            video = moviepy.editor.VideoFileClip(f"{FPATH}/{name}/{i}.{format}")
            if int(video.duration) <= subclip_duration//2:
                raise DurationError
        except:
            print(f"The {i} subclip was not correctly created, deleting...")
            os.remove(f"{FPATH}/{name}/{i}.{format}")
            continue

        #Creating frames
        if verbose:
            print("....Extracting frames....")
        os.makedirs(f"{DEST}/{name+'_'+str(i)}", exist_ok=True)   #Creating directory for Train dataset
        try:
            video_to_frames(f"{FPATH}/{name}/{i}.{format}", f"{DEST}/{name+'_'+str(i)}", overwrite=False, every=frequency)
        except:
            print("Error occured while executing VIDEO_TO_FRAMES")
            os.rmdir(f"{DEST}/{name+'_'+str(i)}/{i}")
            os.rmdir(f"{DEST}/{name+'_'+str(i)}")
            continue

    #Delete directory with subclips
    if name in os.listdir(f"{FPATH}"):   
        shutil.rmtree(f"{FPATH}/{name}")

def movies_preprocess(movie_list, train=True, n_subclips=3, subclip_duration=30, frequency=45, verbose=False):
    for movie_name in movie_list:
        print(f"Preprocessing {movie_name}")
        preprocess(movie_name, train, n_subclips, subclip_duration, frequency, verbose)
        

if __name__ == '__main__':
    markup = pd.read_csv(MARKUP_PATH)
    movie_list = markup['Name'].values
    movies_preprocess(movie_list=movie_list, verbose=True)
