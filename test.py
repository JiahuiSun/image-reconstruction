import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import argparse
from os.path import join as pjoin
import time

from dataset import Mnist
from utils import Logger, convert_state_dict
import models


def main(args, logger):
    # ================ seed and device ===================
    np.random.seed(42)
    torch.manual_seed(42)
    # if args.cuda:
    if True:
        torch.cuda.manual_seed_all(42)
        device = 'cuda'
    else:
        device = 'cpu'
    # ================= data ====================
    mnist = Mnist(args.data_dir, mode='test')
    val_loader = torch.utils.data.DataLoader(mnist.test_dataset, batch_size=args.batch_size, shuffle=False)
    # ================== Load Model ===================
    model = models.get_model(name=args.model, n_class=mnist.n_class) 
    model.to(device)
    best_model_path = pjoin(args.save_dir, args.model+'_best_model.pkl')
    state = convert_state_dict(torch.load(best_model_path)["model_state"])
    model.load_state_dict(state)
    # ================== Testing ======================
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for idx, (img, lab) in enumerate(val_loader):
            img = img.to(device)
            lab = lab.to(device)
            out = model(img)
            pred = out.argmax(dim=1, keepdim=True)
            val_acc += pred.eq(lab.view_as(pred)).sum().item()
    val_acc /= len(val_loader.dataset)
    logger.write(f'Model {best_model_path}, Acc: {val_acc:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Classification')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--data-dir', type=str, default='/home/jjou/sunjiahui/MLproject/dataset/MNIST')
    parser.add_argument('--save-dir', type=str, default='./saved')
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()
    logger = Logger(pjoin(args.save_dir, args.model+'_test.log'))
    logger.write(f'\nConfig: {args}')

    main(args, logger)
