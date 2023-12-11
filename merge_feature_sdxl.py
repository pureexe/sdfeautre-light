# merge feature from different layers into one feature

import os
import torch 
from tqdm.auto import tqdm
from multiprocessing import Pool

def aggragate_feature(features, feature_size=(128, 128)):
    agg_feature = []
    for feature in features:
        up_sample = torch.nn.Upsample(size=feature_size, mode="nearest")
        feat = up_sample(feature.unsqueeze(0)).squeeze()
        agg_feature.append(feat)
    agg_feature = torch.cat(agg_feature, axis=0)
    return agg_feature

def process_file(filename, feature_step=900, root="data/polyhaven", feature_layers=[0, 1, 2]):
    feature_paths = [os.path.join(root, "sdxl_feature", f"t_{feature_step}", f"{filename}_layer{i}.pt") for i in feature_layers]
    features = [torch.load(path, map_location=torch.device('cpu')).to(dtype=torch.float32) for path in feature_paths]
    feature = aggragate_feature(features)
    torch.save(feature, os.path.join(root, "sdxl_feature", f"t_{feature_step}_merge", f"{filename}.pt"))

def main():
    root = "data/polyhaven"
    feature_step = 900
    feature_layers = [0, 1, 2]
    split = "test"
    per_scene = 10 #if split == "train" else 1
    with open(os.path.join(root, f"{split}.txt")) as f:
        files = [a.strip() for a in f.readlines()]
        files = [f"{a}_{i:02d}" for a in files for i in range(per_scene)]
        with Pool(16) as p:
            r = list(tqdm(p.imap(process_file, files), total=len(files)))
            
        # for filename in files:
        #     process_file(filename, feature_step, root, feature_layers)
    
    
if __name__ == "__main__":
    main()