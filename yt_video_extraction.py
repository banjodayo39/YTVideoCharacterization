from pytube import YouTube
from tqdm import tqdm
import asyncio
import time
from moviebarcode import Moviebarcode 

def download_yt_video(videopath, save_dir):
    yt = YouTube(videopath)
    stream = yt.streams.filter(file_extension="mp4")
    save_path = save_dir + "/" + videopath[-10:]
    stream.download(save_path)
    return save_path

async def download_yt_videos(videopaths, save_dir, delay=60):
    for videopath in tqdm(videopaths):
        download_yt_video(videopath, save_dir)
        await asyncio.sleep(delay)


def video_to_barcode(video_path , save_dir, width = 1024, height= 440):
    moviebarcode = Moviebarcode(video_path)
    moviebarcode.generate()
    name = save_dir + video_path.split("/")[-2].split(".")[0] + '.png'
    moviebarcode.make_image(file_name=name)
    return moviebarcode


def get_videobarcode_with_link(video_link, save_dir, width = 1024, height= 440):
    video_path = download_yt_video(video_link)
    moviebarcode = Moviebarcode(video_path)
    moviebarcode.generate()
    name = save_dir + video_path.split("/")[-2].split(".")[0] + '.png'
    moviebarcode.make_image(file_name=name)
    return moviebarcode

async def download_yt_videos(videolinks, save_dir, delay=60, width = 1024, height= 440):
    for video_link in tqdm(videolinks):
        get_videobarcode_with_link(video_link, save_dir, width, height)
        await asyncio.sleep(delay)