
import torch
import os 
import numpy as np

class Feature2SHDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root, 
                 split="train", 
                 light_type="ldr", 
                 lmax=6, 
                 feature_step=900, 
                 feature_layers=[0, 1, 2],
                 feature_size = (128, 128),
                 sh_dir="pysh_100"
                 ):
        super().__init__()
        self.root = root
        self.files = []
        self.light_type = light_type
        self.lmax = lmax
        self.feature_step = feature_step
        self.feature_layers = feature_layers
        self.feature_size = feature_size
        self.sh_dir = sh_dir
        self.split = split
        per_scene = 10 if split.startswith("train") else 1
        with open(os.path.join(root, f"{split}.txt")) as f:
            files = [a.strip() for a in f.readlines()]
            self.files = [f"{a}_{i:02d}" for a in files for i in range(per_scene)]
            
    def aggragate_feature(self, features):
        agg_feature = []
        for feature in features:
            # print(feature.shape, size)
            up_sample = torch.nn.Upsample(size=self.feature_size, mode="nearest")
            feat = up_sample(feature.unsqueeze(0)).squeeze()
            agg_feature.append(feat)
        agg_feature = torch.cat(agg_feature, axis=0)
        return agg_feature
        
    def __getitem__(self, index: int) -> Any:
        filename = self.files[index]
        if self.split.startswith("val"):
            if filename.startswith("test_"):
                filename = filename[5:]
            elif filename.startswith("train_"):
                filename = filename[6:]
        
        # aggregate feature
        feature_paths = [os.path.join(self.root, "sdxl_feature", f"t_{self.feature_step}", f"{filename}_layer{i}.pt") for i in self.feature_layers]
        features = [torch.load(path, map_location=torch.device('cpu')).to(dtype=torch.float32) for path in feature_paths]
        feature = self.aggragate_feature(features)
        
        # read sh
        npy_path = os.path.join(self.root, self.sh_dir, self.light_type, f"{filename}.npy")
        sh_full = np.load(npy_path)
        sh_coeff = flatten_sh_coeff(sh_full, self.lmax)
        sh_coeff = torch.from_numpy(sh_coeff).to(dtype=torch.float32).view(-1)
        
        ret_dict = {
            "name": self.files[index],
            "feature": feature,
            "sh": sh_coeff
        }
        
        if self.split == "val":
            # load with PIL
            ret_dict["image"] = Image.open(os.path.join(self.root, "rectangle", "ldr", f"{filename}.png")).convert("RGB").resize((256,256))
            ret_dict["env_ldr"] = Image.open(os.path.join(self.root, "envmap", "ldr", f"{filename}.png")).convert("RGB").resize((512,256))
            # conver to tensor
            ret_dict["image"] = torchvision.transforms.functional.pil_to_tensor(ret_dict["image"]) / 255.0
            ret_dict["env_ldr"] = torchvision.transforms.functional.pil_to_tensor(ret_dict["env_ldr"]) / 255.0
                        
        return ret_dict
    
    def __len__(self) -> int:
        return len(self.files)