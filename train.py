import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from os.path import join as pjoin
import time
# from tensorboardX import SummaryWriter

from dataset import Mnist
from utils import Logger, AverageMeter, RunningScore
import models


def main(args):
    # ================ seed, device, log ===================
    np.random.seed(42)
    torch.manual_seed(42)
    if args.cuda: 
        torch.cuda.manual_seed_all(42)
    device = 'cuda' if args.cuda else 'cpu'
    lr = int(-np.log10(args.lr))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logdir = pjoin(args.save_dir, args.model, f'lr{lr}', f'{args.optim}', f'patience{args.patience}')
    logger = Logger(pjoin(logdir, f'{stamp}.log'))
    logger.write(f'\nTraining configs: {args}')
    # writer = SummaryWriter(log_dir=logdir)

    # ================= load data ====================
    mnist = Mnist(args.data_dir, mode='train')
    train_loader = torch.utils.data.DataLoader(mnist.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(mnist.val_dataset, batch_size=args.batch_size, shuffle=False)

    # ================== Model setup ===================
    model = models.get_model(name=args.model, n_class=mnist.n_class) 
    model.to(device)

    # ================== Loss, optimizer and scheduler ===============
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                        patience=args.patience, verbose=False, threshold=1e-4)

    # ================== Train model ===============
    start_epoch = 0
    if args.resume:
        logger.write("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint['epoch']
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    for epoch in range(start_epoch, args.n_epoch):
        # ================== Train ====================
        model.train()
        st = time.time()
        for idx, (img, lab) in enumerate(train_loader):
            img = img.to(device)
            lab = lab.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item())
            if args.print_freq and (idx+1)%args.print_freq == 0:
                logger.write(f'Epoch: [{epoch}][{idx+1}/{len(train_loader)}]\t'
                             f'Loss: {loss.item():.4f}\t'
                             f"lr: {optimizer.param_groups[0]['lr']:.6f}\t")

        # ===================== Validation ===================
        model.eval()
        mid = time.time()
        best_acc = -100
        with torch.no_grad():
            for idx, (img, lab) in enumerate(val_loader):
                img = img.to(device)
                lab = lab.to(device)
                out = model(img)
                pred = out.argmax(dim=1, keepdim=True)
                val_acc = pred.eq(lab.view_as(pred)).sum().item() / len(lab)
                loss = criterion(out, lab)
                val_loss_meter.update(loss.item())
                val_acc_meter.update(val_acc)
        end = time.time()

        # =================== adjust lr, save model and print log ================
        scheduler.step(val_acc_meter.avg)
        logger.write(f'Epoch: {epoch:2d} Train Loss: {train_loss_meter.avg:.4f}  Val Loss: {val_loss_meter.avg:.4f}  Val Acc: {val_acc_meter.avg:.4f}')
        logger.write(f'Train cost: {round(mid-st)}s Val cost: {round(end-mid)}s')
        # writer.add_scalar('train/loss', train_loss_meter.avg, epoch)
        # writer.add_scalar("val/loss", val_loss_meter.avg, epoch)
        # writer.add_scalar('val/acc', val_acc_meter.avg, epoch)

        if val_acc_meter.avg > best_acc:
            best_acc = val_acc
            state = {
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            save_path = pjoin(logdir, f"{stamp}.pkl")
            torch.save(state, save_path)
        train_loss_meter.reset()
        val_loss_meter.reset()
        val_acc_meter.reset()
        
    logger.write(f'Training finished, best acc: {best_acc:.3f}') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Classification')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--data-dir', type=str, default='../dataset/MNIST')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--save-dir', type=str, default='saved')
    parser.add_argument('--print-freq', type=int, default=50)

    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--patience', type=int, default=1000)
    args = parser.parse_args()

    main(args)
