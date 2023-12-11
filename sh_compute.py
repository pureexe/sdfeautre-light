# compute spherical harmonic coefficients
from relighting.sh_utils import get_shcoeff
import skimage 
import ezexr 
import os
import numpy as np 
from tqdm.auto import tqdm
from multiprocessing import Pool

LMAX = 100
INPUT_DIR = "data/polyhaven/envmap/hdr"
OUTPUT_DIR = f"data/polyhaven/pysh_{LMAX}/hdr"


def process_image(filename):
    img_path = os.path.join(INPUT_DIR, filename)
    if filename.endswith(".png"):
        out_path = os.path.join(OUTPUT_DIR, filename[:-4]+".npy")
        img = skimage.io.imread(img_path)
        img = skimage.img_as_float32(img)
    elif filename.endswith(".exr"):
        out_path = os.path.join(OUTPUT_DIR, filename[:-4]+".npy")
        img = ezexr.imread(img_path) # value range 0 to 1
    else:
        return None
    img = img[:, :, :3]
    sh = get_shcoeff(img, LMAX)
    np.save(out_path, sh)
    return None

def main():
    files = os.listdir(INPUT_DIR)
    with Pool(16) as p:
        r = list(tqdm(p.imap(process_image, files), total=len(files)))

if __name__ == "__main__":
    main()