import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
sr_baseline_dir = os.path.join("/".join(sys.path[0].split("/")[0:3]), "sr-baselines")
print(sr_baseline_dir)
sys.path.append(sr_baseline_dir)
# import SRCNN model
from srcnn.models import SRCNN

rootdir = '/'.join(sys.path[0].split("/")[0:-1])
sys.path.append(rootdir)
from srcnn_dataset import SRCNNDataset
from PIL import Image
from dataset_util import FastDataLoader, tensor2image
import util
sys.path.append(os.path.join(rootdir, "train_eval_scripts"))
from async_loader import AsynchronousLoader
from imageio import imsave


def infer(model, valid_queue, criterion, logfile):
  loss_tracker = util.AverageMeter()
  psnr_tracker = util.AverageMeter() 

  with torch.no_grad():
    for step, (lr_img, hr_img) in enumerate(valid_queue):
      out = model(lr_img)
      out = torch.clamp(out, 0, 1.0)

      n = hr_img.shape[0]

      loss = criterion(out, hr_img)
      loss_tracker.update(loss.item(), n)

      # compute running psnr
      per_image_mse = (out-hr_img).square().mean(-1).mean(-1).mean(-1)
      per_image_psnr = -10.0*torch.log10(per_image_mse)
      batch_avg_psnr = per_image_psnr.sum(0) / n
      psnr_tracker.update(batch_avg_psnr.item(), n)

  print('validation %e %2.3f', loss_tracker.avg, psnr_tracker.avg)
  with open(logfile, "a+") as f:
    f.write(f'{psnr_tracker.avg}\n')


def train_epoch(model, train_queue, criterion, optimizer):
  model.train()

  loss_tracker = util.AverageMeter()
  psnr_tracker = util.AverageMeter()

  for step, (lr_img, hr_img) in enumerate(train_queue):
    optimizer.zero_grad()

    out = model(lr_img)

    n = hr_img.shape[0]

    loss = criterion(out, hr_img)
    loss.backward()
    optimizer.step()
    loss_tracker.update(loss.item(), n)

    # upsampled_img = tensor2image(out[0].unsqueeze(0)*255)
    # target_img = tensor2image(hr_img[0].unsqueeze(0)*255)
    # imsave("upsampled_img.png", upsampled_img)
    # imsave("target_img.png", target_img)

    # print(lr_img.shape, hr_img.shape)
    # print(f'lr img min, max {torch.min(lr_img)}, {torch.max(lr_img)}')
    # print(f'hr img min, max {torch.min(hr_img)}, {torch.max(hr_img)}')

    # compute running psnr
    per_image_mse = (out-hr_img).square().mean(-1).mean(-1).mean(-1)
    per_image_psnr = -10.0*torch.log10(per_image_mse)
    batch_avg_psnr = per_image_psnr.sum(0) / n
    psnr_tracker.update(batch_avg_psnr.item(), n)

    # print(f'psnr {per_image_psnr}')
    # exit()

    if step % 100 == 0 or step == len(train_queue)-1:
      print(f"train step {step} psnr {psnr_tracker.avg}")

def train(args):
  device = torch.device(f"cuda:{args.gpu}")

  train_data = SRCNNDataset(args.train)
  sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(len(train_data))))

  train_queue = FastDataLoader(
      train_data, batch_size=64,
      sampler=sampler,
      pin_memory=True, num_workers=8)

  train_loader = AsynchronousLoader(train_queue, device)

  val_data = SRCNNDataset(args.val)
  sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(len(val_data))))

  val_queue = FastDataLoader(
      val_data, batch_size=64,
      sampler=sampler,
      pin_memory=True, num_workers=8)

  val_loader = AsynchronousLoader(val_queue, device)

  model = SRCNN().to(device)
  model._initialize()

  last_params = []

  criterion = nn.MSELoss().to(device)
  for n,p in model.named_parameters():
    print(n)
    if "conv3" in n:
      last_params.append({'params': p, 'lr':args.lr/10.0})

  #optimizer = optim.Adam(last_params, lr=args.lr)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  for epoch in range(args.epochs):
    train_epoch(model, train_loader, criterion, optimizer)
    infer(model, val_loader, criterion, args.logfile)
    model_weight_file = os.path.join(args.weights, f"epoch_{epoch}")
    torch.save(model.state_dict(), model_weight_file)
  

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=100, type=int, metavar="N",
                    help="Number of total epochs to run. (default:100)")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate. (default:0.01)")
parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4, 8],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--train", type=str)
parser.add_argument("--val", type=str)
parser.add_argument("--weights", type=str)
parser.add_argument("--seed", type=str)
parser.add_argument("--logfile", type=str)
parser.add_argument("--gpu", type=int)

args = parser.parse_args()
print(args)

if args.seed is None:
  args.seed = random.randint(1, 10000)
print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

train(args)
