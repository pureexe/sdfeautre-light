import os 
import argparse
import ezexr
from envmap import EnvironmentMap, rotation_matrix
import skimage 
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from relighting.tonemapper import TonemapHDR
import numpy as np

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/data2/pakkapon/datasets/polyhaven_exr" ,help='directory that contain the image') #dataset  directory
    parser.add_argument("--output_dir", type=str, default="data/polyhaven" ,help='output_dir') #dataset name or directory 
    parser.add_argument("--split", type=str, default="data/polyhaven/test.txt" ,help='split') #split_path
    parser.add_argument("--rotate_per_scene", type=int, default=10 ,help='how much to rotate per scene') #how much to rotate per scene to do an auxmentation
    parser.add_argument("--crop_size", type=int, default=1024 , help="image crop_size (width)") 
    parser.add_argument("--env_width", type=int, default=256 , help="image crop_size (width)") 
    parser.add_argument("--max_mapping", type=float, default=0.9 ,help='max_mapping value') #max_mapping value
    parser.add_argument("--percentile", type=int, default=99 ,help='percentile value') #percentile value
    parser.add_argument("--gamma", type=float, default=2.4 ,help='gamma value') #gamma value
    return parser

def process_image(args, image_name):
    hdrenv_dir = os.path.join(args.output_dir, "envmap", "hdr")
    ldrenv_dir = os.path.join(args.output_dir, "envmap", "ldr")
    hdrcrop_dir = os.path.join(args.output_dir, "rectangle", "hdr")
    ldrcrop_dir = os.path.join(args.output_dir, "rectangle", "ldr")
    
    
    os.makedirs(hdrenv_dir, exist_ok=True)
    os.makedirs(ldrenv_dir, exist_ok=True)
    os.makedirs(hdrcrop_dir, exist_ok=True)
    os.makedirs(ldrcrop_dir, exist_ok=True)
    
    for i in range(args.rotate_per_scene):
        target_path = os.path.join(ldrcrop_dir, image_name+f"_{i:02d}.png")
        if os.path.exists(target_path):
            continue
        hdr2ldr = TonemapHDR(gamma=args.gamma, percentile=args.percentile, max_mapping=args.max_mapping)
        # process environment map    
        envmap_path = os.path.join(args.dataset, image_name+"_4k.exr")
        hdr_env = EnvironmentMap(envmap_path, 'latlong')
        if args.rotate_per_scene != 1:
            vertical = 0
            horizontal = (i / 10) * 2 * np.pi
            dcm = rotation_matrix(azimuth=-horizontal,elevation=-vertical,roll=0) 
            hdr_env.rotate(dcm)

        hdr_env = hdr_env.resize((args.env_width, args.env_width * 2))
        ezexr.imwrite(os.path.join(hdrenv_dir, image_name+f"_{i:02d}.exr"), hdr_env.data)
        ldr_env, _, _ = hdr2ldr(hdr_env.data)
        ldr_env = skimage.img_as_ubyte(ldr_env)
        skimage.io.imsave(os.path.join(ldrenv_dir, image_name+f"_{i:02d}.png"), ldr_env)


        # process crop
        envmap_path = os.path.join(args.dataset, image_name+"_4k.exr")
        hdr_env = EnvironmentMap(envmap_path, 'latlong')
        if args.rotate_per_scene == 1:
            dcm = rotation_matrix(azimuth=0,elevation=0,roll=0)    
        else:
            vertical = 0
            horizontal = (i / 10) * 2 * np.pi
            dcm = rotation_matrix(azimuth=-horizontal,elevation=-vertical,roll=0) 
            
        hdr_image = hdr_env.project(vfov=60., # degrees
            rotation_matrix=dcm,
            ar=1./1.,
            resolution=(args.crop_size, args.crop_size),
            projection="perspective",
            mode="normal"
        )
        ezexr.imwrite(os.path.join(hdrcrop_dir, image_name+f"_{i:02d}.exr"), hdr_image)
        
        ldr_image, _, _ = hdr2ldr(hdr_image)
        ldr_image = skimage.img_as_ubyte(ldr_image)
        skimage.io.imsave(os.path.join(ldrcrop_dir, image_name+f"_{i:02d}.png"), ldr_image)
    

def main():
    args = create_argparser().parse_args()
    with open(args.split, "r") as f:
        image_names = f.readlines()
    
    image_names = [image_name.strip() for image_name in image_names]
    
    fn = partial(process_image, args)
    
    
    
    
    with Pool(os.cpu_count() // 2) as p:
        r = list(tqdm(p.imap(fn, image_names), total=len(image_names)))
        
    
    
    
if __name__ == "__main__":
    main()