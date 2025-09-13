import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import h5py

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from PIL import Image  
import clip 
import glob 
import hydra
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import cv2
import time
from captum.attr import visualization as viz
from liv import load_liv
from liv.trainer import Trainer
from liv.utils import utils
from liv.utils.data_loaders import LIVBuffer
from liv.utils.logger import Logger
from liv.utils.plotter import plot_reward_curves

import cv2
from captum.attr import IntegratedGradients
from torchvision import transforms


def apply_integrated_gradients(model, image, baseline=None, steps=50):
    """
    Apply Integrated Gradients on a PyTorch model.
    :param model: Pretrained PyTorch model returning a single scalar
    :param image: Input image as a PyTorch tensor
    :param baseline: Baseline image (default: black image)
    :param steps: Number of steps for IG computation
    :return: Colored heatmap as a NumPy array
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.unsqueeze(0).to(device)   # Add batch dimension
    image.requires_grad = True

    if baseline is None:
        baseline = torch.zeros_like(image)

    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(image, baselines=baseline, return_convergence_delta=True, n_steps=steps)
    transposed_attr = np.transpose(attr.squeeze().cpu().detach().numpy(), (1,2,0))
    transposed_image = np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0))
    #attr = attr.squeeze().detach().cpu().numpy()

    # Aggregate across color channels using absolute sum
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    _ = viz.visualize_image_attr(transposed_attr,
                             transposed_image,
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)
    return _ 
   
def visualize_attributions(input_tensor, attributions, figsize=(10, 5)):
    """Helper function to display attributions on the input tensor."""
    # Display the input
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    _ = viz.visualize_image_attr(transposed_attr_ig,
                             transposed_image,
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)

def overlay_heatmap_on_image(heatmap_colored, original_image_tensor, alpha=0.5):
    """
    Overlay the colored heatmap on the original image.
    :param heatmap_colored: Heatmap as a colored NumPy array (H x W x 3)
    :param original_image_tensor: Original image as a PyTorch tensor (C x H x W)
    :param alpha: Blending factor for heatmap
    :return: Overlayed image as a NumPy array
    """
    original_image_np = original_image_tensor.permute(1, 2, 0).cpu().numpy()
    original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
    original_image_np = np.uint8(255 * original_image_np)

    # Resize heatmap to match original image dimensions
    heatmap_colored = cv2.resize(heatmap_colored, (original_image_np.shape[1], original_image_np.shape[0]))

    overlayed_image = cv2.addWeighted(original_image_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_image, heatmap_colored


def plot_reward_curves(
    manifest, tasks, load_video, encoder_model, fig_filename_prefix, animated=False, num_vid=5
):
    for task in tasks:
        try:
            videos = manifest[manifest["text"] == task]
        except:
            videos  = manifest[manifest["narration"] == task]
        for i in range(num_vid):
            m = videos.iloc[i]
            imgs = load_video(m) 
            fig_filename = f"{fig_filename_prefix}_{task}_{i}".replace(" ", "-")
            distances_cur_img, distances_cur_text = calculate_distances(
                encoder_model, imgs, task
            )
            plot_rewards(
                distances_cur_img,
                distances_cur_text,
                imgs,
                task,
                fig_filename,
                animated=animated,
            )








def make_network(cfg):
    model =  hydra.utils.instantiate(cfg)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    if cfg.device == "cpu":
        model = model.module.to(cfg.device)
    return model
def wrap_with_l2_norm(model: nn.Module) -> nn.Module:
    """
    Wraps the given model so that it outputs the L2 norm of its original output.

    Args:
        model (nn.Module): The original PyTorch model.

    Returns:
        nn.Module: A new model that outputs the L2 norm of the original model's output.
    """
    class L2NormWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            output = self.base_model(x)
            l2_norm = torch.norm(output, p=2, dim=1, keepdim=True)  # dim=1 assumes output is (batch_size, features)
            return l2_norm

    return L2NormWrapper(model)    
class PerturbationAttention:
    """
    See https://arxiv.org/pdf/1711.00138.pdf for perturbation-based visualization
    for understanding a control agent.
    """

    def __init__(self, model, image_size=[128, 128], patch_size=[16, 16], device="cpu"):

        self.model = model
        self.patch_size = patch_size
        H, W = image_size
        num_patches = (H * W) // np.prod(patch_size)
        # pre-compute mask
        h, w = patch_size
        nh, nw = H // h, W // w
        mask = (
            torch.eye(num_patches)
            .view(num_patches, num_patches, 1, 1)
            .repeat(1, 1, patch_size[0], patch_size[1])
        )  # (np, np, h, w)
        mask = rearrange(
            mask.view(num_patches, nh, nw, h, w), "a b c d e -> a (b d) (c e)"
        )  # (np, H, W)
        self.mask = mask.to(device).view(1, num_patches, 1, H, W)
        self.num_patches = num_patches
        self.H, self.W = H, W
        self.nh, self.nw = nh, nw

    def __call__(self, rgb):
        #print("RGB")
        #print(rgb)
        # = data["obs"]["agentview_rgb"]  # (B, C, H, W)
        B, C, H, W = rgb.shape

        rgb_ = rgb.unsqueeze(1).repeat(1, self.num_patches, 1, 1, 1)  # (B, np, C, H, W)
        rgb_mean = rgb.mean([2, 3], keepdims=True).unsqueeze(1)  # (B, 1, C, 1, 1)
        rgb_new = (rgb_mean * self.mask) + (1 - self.mask) * rgb_  # (B, np, C, H, W)
        rgb_stack = torch.cat([rgb.unsqueeze(1), rgb_new], 1)  # (B, 1+np, C, H, W)

        rgb_stack = rearrange(rgb_stack, "b n c h w -> (b n) c h w")
        res = self.model(rgb_stack, modality="vision").view(B, self.num_patches + 1, -1)  # (B, 1+np, E)
        
        base = res[:, 0].view(B, 1, -1)
        #print("BASE")
        #print(base)
        others = res[:, 1:].view(B, self.num_patches, -1)
        #print("OTHERS")
        #print(others)
        attn = F.softmax((others - base).pow(2).sum(-1), -1)  # (B, num_patches)
        #print("ATTN")
        #print(attn)
        attn_ = attn.view(B, 1, self.nh, self.nw)
        attn_ = (
            F.interpolate(attn_, size=(self.H, self.W), mode="bilinear")
            .detach()
            .cpu()
            .numpy()
        )
        return attn_

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logging = self.cfg.logging
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        if self.logging:
            self.setup()

        if not cfg.eval:
            print("Creating Dataloader")
            print(self.cfg.datapath_train)
            train_iterable = LIVBuffer(datasource=self.cfg.dataset, datapath=self.cfg.datapath_train, num_workers=self.cfg.num_workers, num_demos=self.cfg.num_demos, doaug=self.cfg.doaug, alpha=self.cfg.alpha)
            self.train_dataset = train_iterable
            self.train_loader = iter(torch.utils.data.DataLoader(train_iterable,
                                            batch_size=self.cfg.batch_size,
                                            num_workers=self.cfg.num_workers,
                                            pin_memory=True))

        ## Init Model
        print("Initializing Model")
        self.model = make_network(cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0

        ## If reloading existing model
        if cfg.load_snap:
            print("LOADING", cfg.load_snap)
            self.load_snapshot(cfg.load_snap)

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=False, cfg=self.cfg)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.train_steps, 1)
        eval_freq = self.cfg.eval_freq
        eval_every_step = utils.Every(eval_freq, 1)

        # trainer = Trainer()
        trainer = hydra.utils.instantiate(self.cfg.trainer)

        ## Training Loop
        print("Begin Training")
        while train_until_step(self.global_step):
            if eval_every_step(self.global_step):
                self.generate_reward_curves()
                self.save_snapshot()
            
            ## Sample Batch
            t0 = time.time()
            batch = next(self.train_loader)
            t1 = time.time()
            metrics, st = trainer.update(self.model, batch, self.global_step)
            t2 = time.time()
            if self.logging:
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if self.global_step % 10 == 0:
                print(self.global_step, metrics)
                print(f'Sample time {t1-t0}, Update time {t2-t1}')
                
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.global_step}.pt'
        global_snapshot =  self.work_dir / f'snapshot.pt'
        sdict = {}
        sdict["liv"] = self.model.module.state_dict()
        sdict["optimizer"] = self.model.module.encoder_opt.state_dict()
        sdict["global_step"] = self._global_step
        torch.save(sdict, snapshot)
        torch.save(sdict, global_snapshot)

    def load_snapshot(self, snapshot_path):
        if snapshot_path != 'liv':
            payload = torch.load(snapshot_path)
            self.model.module.load_state_dict(payload['liv'])
        else:
            self.model = load_liv()
        clip.model.convert_weights(self.model)
        try:
            self._global_step = payload['global_step']
        except:
            print("Warning: No global step found")

    def generate_saliency_map(self):
        self.model.eval()
        print("HI")
        os.makedirs(f"{self.work_dir}/reward_curves", exist_ok=True)
        transform = T.Compose([T.ToTensor()])
        

        if self.cfg.dataset not in ["epickitchen"]:
            manifest = pd.read_csv(os.path.join(self.cfg.datapath_train, "manifest.csv"))
            tasks = manifest["text"].unique()
        else:
            manifest = pd.read_csv(os.path.join(self.cfg.datapath_train, "EPIC_100_validation.csv"))
            tasks = ["open microwave", "open cabinet", "open door"]

        fig_filename = f"{self.work_dir}/reward_curves/{self._global_step}_{self.cfg.dataset}"
        if self.cfg.dataset in ["epickitchen"]:
            def load_video(m):
                imgs_tensor = []
                start_frame = m["start_frame"]
                end_frame = m["stop_frame"]
                vid = f"/data2/jasonyma/EPIC-KITCHENS/frames/{m['participant_id']}/rgb_frames/{m['video_id']}"
                for index in range(start_frame, end_frame):
                    img = Image.open(f"{vid}/frame_0000{index+1:06}.jpg")
                    imgs_tensor.append(transform(img))
                imgs_tensor = torch.stack(imgs_tensor)
                return imgs_tensor

        else:
            def load_video(m):
                imgs_tensor = []
                vid = m["directory"]
                for index in range(m["num_frames"]):
                    try:
                        img = Image.open(f"{vid}/{index}.png")
                    except:
                        img = Image.open(f"{vid}/{index}.jpg")
                    imgs_tensor.append(transform(img))
                imgs_tensor = torch.stack(imgs_tensor)
                return imgs_tensor
            
            
            
            for task in tasks:
                try:
                    videos = manifest[manifest["text"] == task]
                except:
                    videos  = manifest[manifest["narration"] == task]
                #FA = PerturbationAttention(self.model)
                
                for i in range(len(videos)):
                    f = h5py.File('/home/pa1077/LIBERO/libero/datasets/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo_copy.hdf5' , "r+")
                    m = videos.iloc[i]
                   
                    vid = m["directory"]
                    print(vid)
                    print("demo_" + str(i))
                    demo=f["data"]["demo_" + str(i)]
                    #hdf5_path
                    #imgs = load_video(m)
                    for j in range(90):
                    #for j in range(m['num_frames']):
                        
                        #print(f"{vid}/{j}.png")
                        img = Image.open(f"{vid}/{j}.png")
                        #img = Image.open('/home/pa1077/LIBERO/libero/datasets/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo/AGENTVIEW/demo_0/0.png')
                        #img = np.asarray(torch.tensor(FA.__call__(transform(img).unsqueeze(0))).view(128, 128))
                        #img_b = np.stack((img,) * 3, axis=-1)
                        _ = apply_integrated_gradients(wrap_with_l2_norm(self.model), transform(img))
                        _[0].savefig("/home/pa1077/LIV/liv/" + "corrected_FA" + str(j) + ".png")
                       
                        #img = np.asarray(torch.tensor(FA.__call__(transform(img).unsqueeze(0)).view(128, 128) * 255)).astype(np.uint8)
                      
                       
                       #print(FA.__call__(imgs)[0])
                       

                        #print("RENDE")
                        if "AGENTVIEW" in vid and j < 20:
                            #image= Image.fromarray(overlay_heatmap_on_image(ht_mp, transform(img))[0])
                            #image_b= Image.fromarray(overlay_heatmap_on_image(ht_mp, transform(img))[1])
                            #image.save("/home/pa1077/LIV/liv/trialNSUM_OVERLAY" + str(j) + ".png")
                            #image_b.save("/home/pa1077/LIV/liv/ONLY_OVERLAY" + str(j) + ".png")
                            #visualize_attributions(transform(img), attributions)
                            print("VISUALIZE")

                            #demo['obs']['agentview_rgb'][j] = ht_mp
                        else:
                            print("DONE")
                            #demo['obs']['eye_in_hand_rgb'][j] = ht_mp


                        #img = Image.fromarray(np.asarray(torch.tensor(FA.__call__(imgs)[j]).view(128, 128) * 255).astype(np.uint8))
                        #img.save("/home/pa1077/LIV/liv/" + str(j) + ".png")
                    
                
                     
                    
                    
                    


@hydra.main(config_path='cfgs', config_name='config_liv')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)

    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)
    print("TEST") 
    if not cfg.eval:
        workspace.train()
    else:
        print("TESTB")
        workspace.generate_saliency_map()

if __name__ == '__main__':
    main()