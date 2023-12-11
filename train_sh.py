import os
import argparse
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT 
import torch
import torchvision
import lightning as L 
from PIL import Image 
import skimage
import numpy as np 
from relighting.sh_utils import flatten_sh_coeff, compute_background
from relighting.network import SimpleCNN, ConvMixer
import diffusers

#torch.multiprocessing.set_start_method('spawn')


L_MAX = 6

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/data2/pakkapon/relight/sdfeautre-light/data/polyhaven" ,help='dataset path')
    parser.add_argument("--gpus", type=int, default=1 ,help='num gpu')
    parser.add_argument("--coeff_level", type=int, default=6 ,help='how many sh level need')
    parser.add_argument("--split_type", type=str, default="" ,help='select split file')
    parser.add_argument("--learning_rate", type=float, default=1e-3 ,help='learning_rate')
    parser.add_argument("--batch_multiplier", type=int, default=1,help='multiple data in batch size to decease loading time')
    parser.add_argument('--cache_feature', action=argparse.BooleanOptionalAction)
    parser.add_argument('--model_type', type=str, default="covmixer", help="network type")

    parser.add_argument("--epochs", type=int, default=1000,help='number of epoch')
    parser.add_argument("--per_scene", type=int, default=10, help='number of train image per scene')
    parser.add_argument("--output_type", 
        const="sh",
        default="sh",
        type=str,
        nargs='?',
        choices=['sh','envmap'],
        help='how many sh level need'
    )
    return parser

class Feature2SHNetwork(L.LightningModule):
    def __init__(self, 
        learning_rate=1e-3,
        coeff_level=6,
        model_type="covmixer",
        **kwargs
    ):
        super().__init__()
        self.model = None

        if "program_config" in kwargs:
            self.program_config = kwargs['program_config']

        self.save_hyperparameters()
        self.build_model()
        
    def build_model(self):
        if self.hparams.model_type == "covmixer":
            num_sh = (self.hparams.coeff_level+1)**2 * 3 # multiply by 3 for RGB
            self.model = ConvMixer(in_channel=2240, out_channel=num_sh, dim=512, depth=8)
        elif self.hparams.model_type == "unet2d":
            self.model = 

        elif self.hparams.model_type == "simple_cnn":
            self.model = SimpleCNN(2240, modifier=2.0)
        else:
            raise NotImplementedError()
        
    def training_step(self, batch, batch_idx):
        gt_sh = batch["sh"]
        pred_sh = self.model(batch["feature"])
        loss = torch.nn.functional.mse_loss(pred_sh, gt_sh)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):

        if self.global_step == 0 and hasattr(self,'program_config'):
            # write log of argparse
            self.logger.experiment.add_text("program_config", str(self.program_config), self.global_step)


        gt_sh = batch["sh"]
        pred_sh = self.model(batch["feature"])
        loss = torch.nn.functional.mse_loss(pred_sh, gt_sh)

        # plot validation loss
        self.log("val_loss", loss)
        
        # show image differnet
        batch_size = gt_sh.shape[0]
        for i in range(batch_size):
            # compute background
            pred_sh_i = pred_sh[i].detach().cpu().view(3,-1).numpy()
            gt_sh_i = gt_sh[i].detach().cpu().view(3,-1).numpy()
            pred_envmap = compute_background(pred_sh_i, lmax=self.hparams.coeff_level, hfov=0, image_size=256)
            gt_envmap = compute_background(gt_sh_i, lmax=self.hparams.coeff_level, hfov=0, image_size=256)
            real_envmap = batch["env_ldr"][i].detach().cpu().permute(1,2,0).numpy()
            input_image = batch["image"][i].detach().cpu().permute(1,2,0).numpy()
            
            psnr = skimage.metrics.peak_signal_noise_ratio(real_envmap, pred_envmap)
            self.log("val_psnr/{}".format(batch["name"][i]), psnr)
            
            output_image = np.concatenate([input_image, real_envmap, gt_envmap, pred_envmap], axis=1) #range 0 to 1
            output_image = torch.from_numpy(output_image).permute(2,0,1)
            self.logger.experiment.add_image(f"{batch['name'][i]}", output_image, self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
class Feature2SHDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root, 
                 split="train", 
                 light_type="ldr", 
                 lmax=6, 
                 feature_step=900, 
                 feature_layers=[0, 1, 2],
                 feature_size = (128, 128),
                 per_scene=0, # 0 mean auto
                 batch_multiplier=1,
                 is_cache_feature=False,
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
        self.batch_multiplier = batch_multiplier
        self.is_cache_feature = is_cache_feature
        if is_cache_feature:
            self.cache_features = {}
        if per_scene == 0:
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
        
        file_index = index % len(self.files) #support batch multipier
        filename = self.files[file_index]

        if self.split.startswith("val"):
            if filename.startswith("test_"):
                filename = filename[5:]
            elif filename.startswith("train_"):
                filename = filename[6:]
        
        # aggregate feature
        if self.is_cache_feature and filename in self.cache_features:
            # support cache feature to reduce harddisk load but can run only very small dataset
            feature = self.cache_features[filename]
        else:
            feature_paths = [os.path.join(self.root, "sdxl_feature", f"t_{self.feature_step}", f"{filename}_layer{i}.pt") for i in self.feature_layers]
            features = [torch.load(path, map_location=torch.device('cpu')).to(dtype=torch.float32) for path in feature_paths]
            feature = self.aggragate_feature(features)
            if self.is_cache_feature:
                self.cache_features[filename] = feature
        
        # read sh
        npy_path = os.path.join(self.root, self.sh_dir, self.light_type, f"{filename}.npy")
        sh_full = np.load(npy_path)
        sh_coeff = flatten_sh_coeff(sh_full, self.lmax)
        sh_coeff = torch.from_numpy(sh_coeff).to(dtype=torch.float32).view(-1)
        
        ret_dict = {
            "name": self.files[file_index],
            "feature": feature,
            "sh": sh_coeff
        }
        
        if self.split.startswith("val"):
            # load with PIL
            ret_dict["image"] = Image.open(os.path.join(self.root, "rectangle", "ldr", f"{filename}.png")).convert("RGB").resize((256,256))
            ret_dict["env_ldr"] = Image.open(os.path.join(self.root, "envmap", "ldr", f"{filename}.png")).convert("RGB").resize((512,256))
            # conver to tensor
            ret_dict["image"] = torchvision.transforms.functional.pil_to_tensor(ret_dict["image"]) / 255.0
            ret_dict["env_ldr"] = torchvision.transforms.functional.pil_to_tensor(ret_dict["env_ldr"]) / 255.0
                        
        return ret_dict
    
    def __len__(self) -> int:
        return len(self.files) * self.batch_multiplier

def main():
    args = create_argparser().parse_args()
    model = Feature2SHNetwork(
        learning_rate = args.learning_rate,
        coeff_level = args.coeff_level,
        model_type = args.model_type,
        program_config = args
    )
   
    # dataset handling
    train_dataset = Feature2SHDataset(
        root=args.dataset,
        split="train"+args.split_type,
        light_type="ldr",
        lmax=args.coeff_level,
        per_scene=args.per_scene,
        batch_multiplier=args.batch_multiplier,
        is_cache_feature=args.cache_feature,
        feature_step=900
    )
    val_dataset = Feature2SHDataset(
        root=args.dataset,
        split="val"+args.split_type,
        light_type="ldr",
        lmax=args.coeff_level,
        feature_step=900
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=int(os.cpu_count()/2))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=16)

    trainer = L.Trainer(
        accelerator="gpu", 
        devices=args.gpus,
        max_epochs=args.epochs,
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

if __name__ == "__main__":
    main()