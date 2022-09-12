from swinir_wrapper import SuperResolutionModel
import imageio
import numpy as np
import time
from argparse import ArgumentParser
import torch
from skimage import img_as_float32
from first_order_model.reconstruction import *
from first_order_model.utils import get_main_config_params

config = '/data1/pantea/aiortc/nets_implementation/first_order_model/config/paper_configs/exps_overview/swinir/lr256_tgt75Kb.yaml'
video_path = '/data1/pantea/aiortc/swinir_test.mp4'
log_dir = './swinir_api_test'
output_name = 'prediction'

main_configs = get_main_config_params(config)
generator_type = main_configs['generator_type']
use_lr_video = main_configs['use_lr_video']
lr_size = main_configs['lr_size']
print(main_configs)

video_duration = get_video_duration(video_path)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

## model initialization and warm-up
model = SuperResolutionModel(config, 'None')
source_lr = np.random.rand(lr_size, lr_size, model.get_shape()[2])
for _ in range(1):
    _ = model.predict_with_lr_video(source_lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictions = []
container = av.open(file=video_path, format=None, mode='r')
stream = container.streams.video[0]
frame_idx = 0
for av_frame in container.decode(stream):
    frame = av_frame.to_rgb().to_ndarray()
    driving_lr = frame
    prediction = model.predict_with_lr_video(driving_lr)
    predictions.append(prediction)
    if frame_idx % 100 == 0:
        print('total frames', frame_idx)
    frame_idx += 1

imageio.mimsave(os.path.join(log_dir,
                output_name + '.mp4'),
                predictions, fps = 30)

