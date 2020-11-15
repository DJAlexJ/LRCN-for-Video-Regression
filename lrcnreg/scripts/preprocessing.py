#!pip install decord

import numpy as np
import pandas as pd
import cv2
import os
import imageio
import shutil
#from decord import VideoReader
#from decord import cpu, gpu
import moviepy
import moviepy.editor
import shutil
from lrcnreg.config import FPATH, TRAINING_PATH, PREDICTION_PATH, MARKUP_PATH

# imageio.plugins.ffmpeg.download()
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def format_ok(f):
    if f in ['mov', 'mp4', 'flv']:
        return True
    return False


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
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

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, video_filename.split('.')[0], "{:010d}.jpg".format(frame))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved

#def extract_frames_dec(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
#    """
#    Extract frames from a video using decord's VideoReader
#    :param video_path: path of the video
#    :param frames_dir: the directory to save the frames
#    :param overwrite: to overwrite frames that already exist?
#    :param start: start frame
#    :param end: end frame
#    :param every: frame spacing
#    :return: count of images saved
#    """
#
#    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
#    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible
#
#    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path
#
#    assert os.path.exists(video_path)  # assert the video file exists
#
#    # load the VideoReader
#    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)
#                     
#    if start < 0:  # if start isn't specified lets assume 0
#        start = 0
#    if end < 0:  # if end isn't specified assume the end of the video
#        end = len(vr)
#
#    frames_list = list(range(start, end, every))
#    saved_count = 0
#    
#    try:
#        frames = vr.get_batch(frames_list).asnumpy()
#    except Exception as e:
#        print(e)
#
#    for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
#        save_path = os.path.join(frames_dir, video_filename.split('.')[0], "{:010d}.jpg".format(index))  # create the save path
#        if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
#            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # save the extracted image
#            saved_count += 1  # increment our counter by one
#
#    return saved_count  # and return the count of the images we saved


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
    
    name = '.'.join(movie_name.split('.')[:-1])
    format_ = movie_name.split('.')[-1]
    
    if not format_ok(format_):
        print('Skipping file, which is not a video...')
        return
        
    if train == True:
        if not os.path.isdir(TRAINING_PATH):
            os.mkdir(TRAINING_PATH)
        DEST = TRAINING_PATH
    else:
        if not os.path.isdir(PREDICTION_PATH):
            os.mkdir(PREDICTION_PATH)
        DEST = PREDICTION_PATH
        
    
    if format_ == 'flv':   #Decord does not work with flv format
        format_ = 'mov'

    #Extracting subclip from trailer
    base = 10

    os.makedirs(f"{FPATH}/{name}", exist_ok=True)
    for i in range(n_subclips): 
        if verbose:
            print(f"{i} iteration...")
            print("....Making subclip....")
        try:
            ffmpeg_extract_subclip(f"{FPATH}/{movie_name}", base, base+subclip_duration, targetname=f"{FPATH}/{name}/{i}.{format_}")
            base = base + subclip_duration
        except BaseException:
            print(f"Some error occured during {i+1} extraction")
            continue

        #Check if all subclips were correctly created
        try:
            video = moviepy.editor.VideoFileClip(f"{FPATH}/{name}/{i}.{format_}")
            if int(video.duration) <= subclip_duration//2:
                raise DurationError
        except:
            print(f"The {i} subclip was not correctly created, deleting...")
            os.remove(f"{FPATH}/{name}/{i}.{format_}")
            continue

        #Creating frames
        if verbose:
            print("....Extracting frames....")
        os.makedirs(f"{DEST}/{name+'_'+str(i)}", exist_ok=True)   #Creating directory for Train dataset
        try:
            video_to_frames(f"{FPATH}/{name}/{i}.{format_}", f"{DEST}/{name+'_'+str(i)}", overwrite=False, every=frequency)
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
