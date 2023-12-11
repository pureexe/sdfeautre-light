import os 
from tqdm.auto import tqdm
import skimage 
from multiprocessing import Pool

DIR =  "data/polyhaven/rectangle/ldr"

def process_image(name):
    img = skimage.io.imread(os.path.join(DIR, name))
    if img.shape[2] == 4:
        img = img[:, :, :3]
        skimage.io.imsave(os.path.join(DIR, name), img)
    return None

def main():

    files = os.listdir(DIR)
    with Pool(16) as p:
        r = list(tqdm(p.imap(process_image, files), total=len(files)))
        

if __name__ == "__main__":
    main()