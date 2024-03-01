import os 
import numpy as np 
import skimage
from tqdm.auto import tqdm
from multiprocessing import Pool

RECTANGLE_DIR = "data/polyhaven/rectangle/ldr"
BALLBG_DIR = "data/polyhaven/ball_withbg/ldr"
BALL_DIR = "data/polyhaven/ball/ldr"

def process_file(file_name):
    if file_name.endswith(".png"):
        mask = np.load("mask_ball256.npy")[...,None]
        mask = mask.astype(np.float32)
        bg = skimage.io.imread(os.path.join(RECTANGLE_DIR, file_name))
        bg = skimage.img_as_float(bg)[...,:3]
        bg = skimage.transform.resize(bg, (256,256))
        ball = skimage.io.imread(os.path.join(BALL_DIR, file_name))
        ball = skimage.img_as_float(ball)[...,:3]
        img = bg.copy()
        img = ball * mask + bg * (1 - mask)
        img = skimage.img_as_ubyte(img)
        skimage.io.imsave(os.path.join(BALLBG_DIR, file_name), img)

def main():

    # using multiprocessing and tqdm to run process_file in parallel
    with Pool(8) as p:
        list(tqdm(p.imap(process_file, os.listdir(RECTANGLE_DIR)), total=len(os.listdir(RECTANGLE_DIR))))

    # for file_name in tqdm(os.listdir(RECTANGLE_DIR)):
    #     if file_name.endswith(".png"):
    #         bg = skimage.io.imread(os.path.join(RECTANGLE_DIR, file_name))
    #         bg = skimage.img_as_float(bg)[...,:3]
    #         ball = skimage.io.imread(os.path.join(BALL_DIR, file_name))
    #         ball = skimage.img_as_float(ball)[...,:3]
    #         img = bg.copy()
    #         img[512-128:512+128, 512-128:512+128] = ball * mask + bg[512-128:512+128, 512-128:512+128] * (1 - mask)
    #         img = skimage.img_as_ubyte(img)
    #         skimage.io.imsave(os.path.join(BALLBG_DIR, file_name), img)


if __name__ == "__main__":
    main()