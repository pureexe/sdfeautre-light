import os
import argparse
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT 
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torchvision
import lightning as L 
from PIL import Image 
import skimage
import numpy as np 
from relighting.sh_utils import flatten_sh_coeff, compute_background
from relighting.network import SimpleCNN, ConvMixer
import diffusers
from diffusers import AutoencoderKL
from lightning.pytorch import seed_everything

#torch.multiprocessing.set_start_method('spawn')


L_MAX = 6

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="", help="experiment name")
    parser.add_argument("--dataset", type=str, default="/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sdfeautre-light/data/polyhaven" ,help='dataset path')
    parser.add_argument("--gpus", type=int, default=1 ,help='num gpu')
    parser.add_argument("--coeff_level", type=int, default=6 ,help='how many sh level need')
    parser.add_argument("--split_type", type=str, default="" ,help='select split file')
    parser.add_argument("--learning_rate", type=float, default=1e-4 ,help='learning_rate')
    parser.add_argument("--batch_multiplier", type=int, default=1,help='multiple data in batch size to decease loading time')
    parser.add_argument('--cache_feature', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no_batchnorm', dest='has_batchnorm', action='store_false', help='use a model without batchnorm')
    parser.set_defaults(has_batchnorm=True)
    parser.add_argument('--no_batchnorm_track', dest='use_batchnorm_track', action='store_false', help='disable stat track for batch norm')
    parser.set_defaults(use_batchnorm_track=True)
    parser.add_argument("--batchnorm_momentum", type=float, default=0.1 ,help='batchnorm_momentum')


    parser.add_argument('--model_type', type=str, default="covmixer", help="network type")
    parser.add_argument('--input_type', type=str, default="envmap", help="Input type (envmap, chromeball)")
    parser.add_argument('--loss_type', type=str, default="mse", help="loss_type")
    parser.add_argument('--convmix_depth', type=int, default=8, help="convmix depth")
    
    parser.add_argument("--epochs", type=int, default=1000,help='number of epoch')
    parser.add_argument("--per_scene", type=int, default=10, help='number of train image per scene')
    parser.add_argument("--batch_size", type=int, default=4, help='number of batch size')
    parser.add_argument('--note', type=str, default="", help="note")
    return parser



class Feature2SHNetwork(L.LightningModule):
    def __init__(self, 
        learning_rate=1e-3,
        coeff_level=6,
        model_type="covmixer",
        loss_type="mse",
        input_type="envmap",
        convmix_depth=8,
        has_batchnorm=True,
        batchnorm_momentum=0.1,
        use_batchnorm_track=True,
        **kwargs
    ):
        super().__init__()
        self.model = None
        if "program_config" in kwargs:
            self.program_config = kwargs['program_config']

        self.save_hyperparameters()
        self.build_model()
        self.is_img2img = self.hparams.model_type in ['unet']
        
    def build_model(self):
        if self.hparams.model_type == "covmixer":
            num_sh = (self.hparams.coeff_level+1)**2 * 3 # multiply by 3 for RGB
            self.model = ConvMixer(
                in_channel=2240, 
                out_channel=num_sh, 
                dim=512, 
                depth=self.hparams.convmix_depth, 
                has_batchnorm=self.hparams.has_batchnorm,
                batchnorm_momentum=self.hparams.batchnorm_momentum,
                use_batchnorm_track=self.hparams.use_batchnorm_track
            )
        elif self.hparams.model_type == "unet":
            print("=====================================")
            print("Model type: Diffuser UNet")
            print("=====================================")
            unet_out = 4 if self.hparams.input_type == "chromeball" else 3
            self.model = diffusers.UNet2DModel(
                in_channels=2240,
                out_channels=unet_out,
                block_out_channels=[64, 128, 192, 256],
            )
            self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
            # disable VAE gradient
            for p in self.vae.parameters():
                p.requires_grad=False


        elif self.hparams.model_type == "simple_cnn":
            num_sh = (self.hparams.coeff_level+1)**2 * 3 # multiply by 3 for RGB
            self.model = SimpleCNN(2240, num_sh, modifier=2.0)
        else:
            raise NotImplementedError()
        
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1) #[1,3,1024,1024]
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.float()
        return image

    def loss(self, pred, gt):
        if self.hparams.loss_type == "mse":
            return torch.nn.functional.mse_loss(pred, gt)
        elif self.hparams.loss_type == "l1":
            return torch.nn.functional.l1_loss(pred, gt)
        else:
            raise NotImplementedError()
        
    def training_step(self, batch, batch_idx):
        if self.is_img2img:
            return self.train_step_envmap(batch, batch_idx)
        else:
            return self.train_step_sh(batch, batch_idx)
    
    def train_step_sh(self, batch, batch_idx):
        gt_sh = batch["sh"]
        
        pred_sh = self.model(batch["feature"])
        loss = self.loss(pred_sh, gt_sh)
        #loss = self.loss(pred_sh[:,0], gt_sh[:,0])
        self.log("train_loss", loss)
        return loss

    def train_step_envmap(self, batch, batch_idx):
        gt = batch["gt"]
        pred = self.model(batch["feature"], timestep=0).sample 
        if self.hparams.input_type == "envmap":
            pred = torchvision.transforms.functional.resize(pred, size=(128,256))
        elif self.hparams.input_type == "chromeball":
            pred = self.decode_latents(pred)
        loss = self.loss(pred, gt)
        self.log("train_loss", loss)
        return loss


    def validation_step_sh(self, batch, batch_idx):
        gt_sh = batch["sh"] #[1,147]
        pred_sh = self.model(batch["feature"])
        loss = self.loss(pred_sh, gt_sh)
        #loss = self.loss(pred_sh[:,0], gt_sh[:,0])
        #print(pred_sh[0,0], gt_sh[0,0], torch.abs(pred_sh[0,0] - gt_sh[0,0]))

        # plot validation loss
        self.log("val_loss", loss)
        
        # show image differnet
        batch_size = gt_sh.shape[0]
        for i in range(batch_size):
            # compute background
            pred_sh_i = pred_sh[i].detach().cpu().view(3,-1).numpy()
            gt_sh_i = gt_sh[i].detach().cpu().view(3,-1).numpy()

            # for loop print diff value
            diff = torch.abs(pred_sh[i] - gt_sh[i]).detach().cpu()
            self.log(f"val_diff_{i:03d}_max", diff.max())
            self.log(f"val_diff_{i:03d}_min", diff.min())
            self.log(f"val_diff_{i:03d}_mean", diff.mean())
            for jdx in range(diff.shape[0]):
                self.log(f"val_diff_{i:03d}/{jdx:03d}", diff[jdx])

            pred_envmap = compute_background(pred_sh_i, lmax=self.hparams.coeff_level, hfov=0, image_size=256)
            gt_envmap = compute_background(gt_sh_i, lmax=self.hparams.coeff_level, hfov=0, image_size=256)
            real_envmap = batch["env_ldr"][i].detach().cpu().permute(1,2,0).numpy()
            input_image = batch["image"][i].detach().cpu().permute(1,2,0).numpy()
            
            psnr = skimage.metrics.peak_signal_noise_ratio(real_envmap, pred_envmap)
            self.log("val_psnr/{}".format(batch["name"][i]), psnr)
            
            output_image = np.concatenate([input_image, real_envmap, gt_envmap, pred_envmap], axis=1) #range 0 to 1
            output_image = torch.from_numpy(output_image).permute(2,0,1)
            self.logger.experiment.add_image(f"{batch['name'][i]}", output_image, self.global_step)

    
    def validation_step_envmap(self, batch, batch_idx):
        gt = batch["gt"]
        pred = self.model(batch["feature"], timestep=0).sample
        if self.hparams.input_type == "envmap":
            pred = torchvision.transforms.functional.resize(pred, size=(128,256))
        elif self.hparams.input_type == "chromeball":
            pred = self.decode_latents(pred)
        loss = self.loss(pred, gt)
        self.log("val_loss", loss)

        batch_size = gt.shape[0]
        for i in range(batch_size):
            # compute background
            pred_i = pred[i].detach().cpu()
            gt_i = gt[i].detach().cpu()
            input_image = batch["image"][i].detach().cpu()

            if self.hparams.input_type == "envmap":
                pred_i = (pred_i + 1.0) / 2.0
                gt_i = (gt_i + 1.0) / 2.0
                input_image = (input_image + 1.0) / 2.0

                pred_i = torch.clamp(pred_i, 0.0, 1.0)
                gt_i = torch.clamp(gt_i, 0.0, 1.0)
                input_image = torch.clamp(input_image, 0.0, 1.0)

            
            psnr = skimage.metrics.peak_signal_noise_ratio(gt_i.permute(1,2,0).numpy(), pred_i.permute(1,2,0).numpy())
            self.log("val_psnr/{}".format(batch["name"][i]), psnr)
            
            output_image = torch.cat([input_image, gt_i, pred_i], axis=-1)
            self.logger.experiment.add_image(f"{batch['name'][i]}", output_image, self.global_step)



    def validation_step(self, batch, batch_idx):

        if self.global_step == 0 and hasattr(self,'program_config'):
            # write log of argparse
            self.logger.experiment.add_text("program_config", str(self.program_config), self.global_step)

        if self.is_img2img:
            return self.validation_step_envmap(batch, batch_idx)
        else:
            return self.validation_step_sh(batch, batch_idx)
        

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
                 sh_dir="pysh_100",
                 model_type="convmixer",
                 input_type="envmap"
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
        self.is_img2img = model_type in ['unet']
        self.input_type = input_type
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

        ret_dict = {
            "name": self.files[file_index],
            "feature": feature,
        }
       
        # read sh
        if self.is_img2img:
            if self.input_type == "envmap":
                ret_dict['gt'] = Image.open(os.path.join(self.root, "envmap", "ldr", f"{filename}.png")).convert("RGB").resize((256,128))
                ret_dict["gt"] = torchvision.transforms.functional.pil_to_tensor(ret_dict["gt"])
                ret_dict["gt"] = ret_dict["gt"] / 255.0
                ret_dict["gt"] = (ret_dict["gt"] *2.0) - 1.0
            elif self.input_type == "chromeball":
                ret_dict["gt"] = Image.open(os.path.join(self.root, "ball_withbg", "ldr", f"{filename}.png")).convert("RGB").resize((1024,1024))
                ret_dict["gt"] = torchvision.transforms.functional.pil_to_tensor(ret_dict["gt"])
                ret_dict["gt"] = ret_dict["gt"] / 255.0
                #ret_dict["gt"] = (ret_dict["gt"] *2.0) - 1.0 (VAE is now using 0-1)
            if self.split.startswith("val"):
                ret_dict["image"] = Image.open(os.path.join(self.root, "rectangle", "ldr", f"{filename}.png")).convert("RGB")
                if self.input_type == "chromeball":
                    ret_dict["image"] = ret_dict["image"].resize((1024, 1024))
                else:
                    ret_dict["image"] = ret_dict["image"].resize((128,128))
                ret_dict["image"] = torchvision.transforms.functional.pil_to_tensor(ret_dict["image"])
                ret_dict["image"] = ret_dict["image"] / 255.0
                if self.input_type != "chromeball":
                    ret_dict["image"] = (ret_dict["image"] *2.0) - 1.0
        else:
            npy_path = os.path.join(self.root, self.sh_dir, self.light_type, f"{filename}.npy")
            sh_full = np.load(npy_path)
            sh_coeff = flatten_sh_coeff(sh_full, self.lmax)
            sh_coeff = torch.from_numpy(sh_coeff).to(dtype=torch.float32).view(-1)
            ret_dict['sh'] = sh_coeff
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
    seed_everything(42)
    args = create_argparser().parse_args()
    model = Feature2SHNetwork(
        learning_rate = args.learning_rate,
        coeff_level = args.coeff_level,
        model_type = args.model_type,
        loss_type = args.loss_type,
        input_type = args.input_type,
        convmix_depth = args.convmix_depth,
        has_batchnorm = args.has_batchnorm,
        batchnorm_momentum = args.batchnorm_momentum,
        use_batchnorm_track = args.use_batchnorm_track,
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
        feature_step=900,
        model_type=args.model_type,
        input_type=args.input_type
    )
    val_dataset = Feature2SHDataset(
        root=args.dataset,
        split="val"+args.split_type,
        light_type="ldr",
        lmax=args.coeff_level,
        feature_step=900,
        model_type=args.model_type,
        input_type=args.input_type
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=4)

    precision = 32 
    if args.model_type == "unet" and args.input_type == "chromeball":
        precision = 16

    save_dir = os.path.join("lightning_logs", args.name) if args.name != "" else os.getcwd() 
    logger = TensorBoardLogger(
        save_dir=save_dir,
    )
    trainer = L.Trainer(
        logger = logger,
        accelerator="gpu", 
        devices=args.gpus,
        max_epochs=args.epochs,
        precision=precision,
        log_every_n_steps=1
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

if __name__ == "__main__":
    main()