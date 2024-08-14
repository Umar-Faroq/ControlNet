# code mostly used from gradio_seg2image.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#from share import *

import cv2
import einops
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import math

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# samples_row represents how many samples you want in a row
def sample_img(input_image, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, scale, seed, samples_row):
    with torch.no_grad():
        input_image = HWC3(input_image)
        #detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(input_image, 512)
        H, W, C = img.shape

        control = torch.from_numpy(input_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # uncomment below if low vram
        #if config.save_memory:
            #model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        #if config.save_memory:
            #model.low_vram_shift(is_diffusing=True)
        
        strength = 1
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta = 0,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        #if config.save_memory:
            #model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
        
        # NEW: Code to view samples in matplotlib grid
        print(results[0].shape)
        num_rows = math.ceil(num_samples / samples_row) + 1
        fig, ax = plt.subplots(num_rows, samples_row)
        ind = -1
        
        # First row is for gt and mask
        # samples_row should be at least 2
        # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html
        # https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

        ax[0,0].imshow(input_image) 
        ax[0,0].set_title("Mask")
        ax[0,1].imshow(target_1)
        ax[0,1].set_title("Ground Truth")

        # Plot the generated images
        for i in range(num_rows):
            for j in range(samples_row):
                ax[i,j].axis("off") 
                if (i != 0 and ind != num_samples-1):
                    ax[i,j].imshow(results[(ind:=ind+1)])
                    ax[i,j].set_title("Sample")
                else:
                    pass

        plt.show()
        
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/version_5/checkpoints/epoch=31-step=14999.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


# Code for sampling

prompt = "A cardiac ultrasound image, in a two chamber view, at the end of systole"  
guidance_scale = 5
samples_row = 4 # how many samples per row

# mask
source_1 = cv2.resize(cv2.imread("./inference/CAMUS/masks/patient0483_2CH_ES_gt.nii.gz.png"), (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/
# gt
target_1 = cv2.resize(cv2.imread("./inference/CAMUS/gt/patient0483_2CH_ES.nii.gz.png"), (512,512), interpolation= cv2.INTER_LINEAR)

sample_img(source_1, prompt, "", "", 8, 50, False, guidance_scale, 1, samples_row)