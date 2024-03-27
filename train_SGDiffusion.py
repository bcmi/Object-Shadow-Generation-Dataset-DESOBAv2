import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import TrainDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
import argparse
import os

def get_args_parser():
    parser = argparse.ArgumentParser('train shadow diffusion', add_help=False)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--logger_freq', default=300, type=int)
    parser.add_argument('--sd_locked', action='store_true', default=True)
    parser.add_argument('--only_mid_control', action='store_true', default=False)
    parser.add_argument('--resume_path', default='data/ckpt/DESOBAv2.ckpt', type=str)
    parser.add_argument('--gpu_id', default=2, type=int)
    parser.add_argument('--train_dataset_path', default='data/desoba_v2', type=str)
    return parser



parser = get_args_parser()
args = parser.parse_args()
torch.cuda.set_device(args.gpu_id)

model = create_model('./models/cldm_v15.yaml').cpu()
if os.path.exists(args.resume_path):
    model_weight = load_state_dict(args.resume_path, location='cpu')
    model.load_state_dict(model_weight, strict=False)
model.learning_rate = args.learning_rate
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control

dataset = TrainDataset(data_file_path=args.train_dataset_path,device=torch.device("cuda:"+str(args.gpu_id)))
dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=args.logger_freq)

trainer = pl.Trainer(gpus=[args.gpu_id], precision=32, callbacks=[logger])
trainer.fit(model, dataloader)
