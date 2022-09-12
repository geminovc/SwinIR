import yaml
import os
from PIL import Image

import torch
from torchvision import transforms
import numpy as np

from options import get_default_args
from main_test_swinir import define_model, test

class SuperResolutionModel():
    def __init__(self, config_path, checkpoint='None'):
        super(SuperResolutionModel, self).__init__()
        if type(config_path) is dict:
            config = config_path
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)

        # config parameters
        generator_params = config['model_params']['generator_params']
        self.shape = config['dataset_params']['frame_shape']
        self.use_lr_video = generator_params.get('use_lr_video', True)
        self.lr_size = generator_params.get('lr_size', 256)
        self.generator_type = generator_params.get('generator_type', 'SwinIR')
        self.scale = int(self.shape[1] / self.lr_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = get_default_args()
        self.args.scale = self.scale
        self.args.task = 'classical_sr'
        self.args.model_path = f'/video-conf/scratch/pantea_mapmaker/SwinIR_checkpoints/001_classicalSR_DF2K_s64w8_SwinIR-M_x{self.scale}.pth'
        
        self.model = define_model(self.args)
        self.model.eval()
        self.model = self.model.to(self.device)


    def get_shape(self):
        return tuple(self.shape)


    def get_lr_video_info(self):
        return self.use_lr_video, self.lr_size


    def reset(self):
        self.times = []

    def prepare_image(self, args, img):
        # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
        # 003 real-world image sr (load lq image only)
        if args.task in ['classical_sr', 'lightweight_sr', 'real_sr']:
            img_lq = img.astype(np.float32) / 255.
        return img_lq


    def predict_with_lr_video(self, target_lr):
        """ predict and return the target RGB frame
            from a low-res version of it.
        """
        img_lq = self.prepare_image(self.args, target_lr) # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        with torch.no_grad():
            # pad input image to be a multiple of window_size
            border = self.args.scale
            window_size = 8
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, self.model, self.args, window_size)
            output = output[..., :h_old * self.args.scale, :w_old * self.args.scale]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output

