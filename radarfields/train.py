from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import glob
import tqdm

class Trainer(object):
    def _init_(self,
                args,
                model,
                split,
                criterion,
                optimizer,
                lr_scheduler,
                device = None,
                max_keep_ckpt = 2,
                scheduler_update_every_step = True,
                skip_ckpt = False
            ):
        self.args = args
        self.name = str(args.name).strip("\"") #保存的checkpoint文件名
        self.training = split == "train"
        self.workspace = Path(args.workspace) #log和checkpoint的路径
        self.max_keep_ckpt = max_keep_ckpt #保存的checkpoint个数
        self.which_checkpoint = args.ckpt
        self.refine_poses = self.args.refine_poses
        self.skip_ckpt = skip_ckpt
        self.learned_norm = self.args.learned_norm
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (device if device else torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))
        self.log_ptr =None
        
        model.to(self.device)
        self.model = model
        self.criterion = criterion
        
        if optimizer is None:
            self.optinmzer = torch.optim.Adam(
                self.model.get_params(arg.lr), betas = (0.9, 0.99), eps = 1e-15
            )
        else:
            self.optinmzer = optimizer
        
        
        