import torch
import torch.nn as nn
import torchvision.transforms as T
from network import GcNet
from read_data import ToTensor, Normalize, Pad
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import cv2

max_disp = 192
left_path = 'test/left/000000_10.png'
right_path = 'test/right/000000_10.png'
model_path = 'test/model/198.ckpt'
save_path = 'test/pic'

mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))
h=256
w=512
maxdisp=160

def main():
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    pairs = {'left': left , 'right' : right}
    transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    pairs = transform(pairs)
    left = pairs['left'].to(device).unsqueeze(0)
    right = pairs['right'].to(device).unsqueeze(0)

    model = GcNet(h , w , maxdisp).to(device)

    state = torch.load(model_path)
    if len(device_ids) == 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            namekey = k[7:] # remove `module.`
            new_state_dict[namekey] = v
        state['state_dict'] = new_state_dict

    model.load_state_dict(state['state_dict'])
    print('load model from {}'.format(model_path))
    print('epoch: {}'.format(state['epoch']))
    print('3px-error: {}%'.format(state['error']))

    model.eval()
    with torch.no_grad():
        disp = model(left, right)

    disp = disp.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(12.84, 3.84))
    plt.axis('off')
    plt.imshow(disp)
    plt.colorbar()
    plt.savefig(save_path, dpi=100)

    print('save diparity map in {}'.format(save_path))


if __name__ == '__main__':
    main()
