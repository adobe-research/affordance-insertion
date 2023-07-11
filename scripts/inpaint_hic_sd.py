import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

from pathlib import Path
from cleanfid import fid
import shutil

import copy
from einops import rearrange, repeat
import scipy.ndimage as ndimage

import wandb
wandb.init(project="affordance")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        '--blend', 
        action='store_true',
        help="blend the pred image with masked image"
    )
    opt = parser.parse_args()

    config = OmegaConf.load('models/affordance/config.yaml')
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/affordance/model.ckpt")["state_dict"], strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    config['data']['params']['batch_size'] = 32
    config['data']['params']['validation']['params']['config']['mode'] = 'swap'
    config['data']['params']['validation']['params']['path'] = './data/'
    data = instantiate_from_config(config.data)
    data.setup()
    dataset = data._val_dataloader()

    with torch.no_grad():
        with model.ema_scope():
            for batch_idx, batch in tqdm(enumerate(dataset)):
                for k in batch:
                    x = batch[k]
                    # x = x[None]
                    if len(x.shape) != 1:   
                        if len(x.shape) == 3:
                            x = x[..., None]
                        x = rearrange(x, 'b h w c -> b c h w')
                    # x = torch.from_numpy(x)
                    x = x.to(memory_format=torch.contiguous_format).float().to(device)
                    batch[k] = x

                all_samples = []
                for cfg_scale in [4.0, ]:
                    ### conditional person insertion sampling
                    for idx in range(3):
                        cond_dict_sdm = {'concat': {}, 'crossattn': {}}
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["masked_image"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"])
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["refer_person_clip"])
                        cond_dict_sdm['uncond_mask'] = batch["uncond_mask"]
                        c_sdm = copy.deepcopy(cond_dict_sdm)
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["zero_person"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"] * 0 + 1)
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["zero_person_clip"])
                        cond_dict_sdm['uncond_mask'] = batch['uncond_mask'] * 0 + 1
                        uc_sdm = copy.deepcopy(cond_dict_sdm)

                        shape_sdm = (4,)+c_sdm['concat']['masked_image'].shape[2:]             

                        c_cond = c_sdm
                        uc_cond = uc_sdm
                        batch_size = c_sdm['concat']['masked_image'].shape[0]                        
                        shape = shape_sdm
                        samples_ddim, intermediates = sampler.sample(S=opt.steps,
                                                        conditioning=c_cond,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        unconditional_guidance_scale=cfg_scale,
                                                        unconditional_conditioning=uc_cond,
                                                        verbose=False)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        all_samples.append(x_samples_ddim)         
                        
                    ### person hallucination sampling
                    for idx in range(3):
                        cond_dict_sdm = {'concat': {}, 'crossattn': {}}
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["masked_image"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"])
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["zero_person_clip"])
                        cond_dict_sdm['uncond_mask'] = batch["uncond_mask"]
                        c_sdm = copy.deepcopy(cond_dict_sdm)
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["zero_person"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"] * 0 + 1)
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["zero_person_clip"])
                        cond_dict_sdm['uncond_mask'] = batch['uncond_mask'] * 0 + 1
                        uc_sdm = copy.deepcopy(cond_dict_sdm)

                        shape_sdm = (4,)+c_sdm['concat']['masked_image'].shape[2:]             

                        c_cond = c_sdm
                        uc_cond = uc_sdm
                        batch_size = c_sdm['concat']['masked_image'].shape[0]                        
                        shape = shape_sdm
                        samples_ddim, intermediates = sampler.sample(S=opt.steps,
                                                        conditioning=c_cond,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        unconditional_guidance_scale=cfg_scale,
                                                        unconditional_conditioning=uc_cond,
                                                        verbose=False)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        all_samples.append(x_samples_ddim)    

                    ### scene hallucination sampling
                    for idx in range(3):
                        cond_dict_sdm = {'concat': {}, 'crossattn': {}}
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["zero_person"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"] * 0 + 1)
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["refer_person_clip"])
                        cond_dict_sdm['uncond_mask'] = batch["uncond_mask"]
                        c_sdm = copy.deepcopy(cond_dict_sdm)
                        cond_dict_sdm['concat']['masked_image'] = model.first_stage_model.encode(batch["zero_person"]).mode()
                        cond_dict_sdm['concat']['mask'] = model.rescale_stage_model.encode(batch["mask"] * 0 + 1)
                        cond_dict_sdm['crossattn']['refer_person'] = model.cond_stage_model.encode(batch["zero_person_clip"])
                        cond_dict_sdm['uncond_mask'] = batch['uncond_mask'] * 0 + 1
                        uc_sdm = copy.deepcopy(cond_dict_sdm)

                        shape_sdm = (4,)+c_sdm['concat']['masked_image'].shape[2:]             

                        c_cond = c_sdm
                        uc_cond = uc_sdm
                        batch_size = c_sdm['concat']['masked_image'].shape[0]                        
                        shape = shape_sdm
                        samples_ddim, intermediates = sampler.sample(S=opt.steps,
                                                        conditioning=c_cond,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        unconditional_guidance_scale=cfg_scale,
                                                        unconditional_conditioning=uc_cond,
                                                        verbose=False)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        all_samples.append(x_samples_ddim)   

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                masked_image = torch.clamp((batch["masked_image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                refer_person = torch.clamp((batch["refer_person"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp(batch["mask"], min=0.0, max=1.0).cpu().numpy().transpose(0,2,3,1)

                def _unnorm(x):
                    return torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)

                cfg_predicted_images = [_unnorm(x) for x in all_samples]

                batch_size = batch['image'].shape[0]
                for elem_idx in range(batch_size):
                    # inpainted = predicted_image[elem_idx]
                    # if opt.blend:
                    #     inpainted = (1-mask[elem_idx])*image[elem_idx]+mask[elem_idx]*predicted_image[elem_idx]
                    # inpainted = inpainted.cpu().numpy().transpose(1,2,0)*255
                    # inpainted = Image.fromarray(inpainted.astype(np.uint8))
                    def proc(img):
                        img = img.cpu().numpy().transpose(1,2,0)*255
                        return Image.fromarray(img.astype(np.uint8))

                    all_imgs = [proc(masked_image[elem_idx]), \
                        proc(refer_person[elem_idx]), \
                        proc(image[elem_idx])]

                    for idx in range(len(cfg_predicted_images)):
                        temp_img = proc(cfg_predicted_images[idx][elem_idx])
                        all_imgs.append(temp_img)

                    wandb.log({"cond_inpainting": [wandb.Image(x) for x in all_imgs]}, \
                        step=batch_idx * batch_size + elem_idx)

