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
            self.optimzer = torch.optim.Adam(
                self.model.get_params(args.lr), betas = (0.9, 0.99), eps = 1e-15
            )
        else:
            self.optimzer = optimizer
        
        # 是否对offset和scaler两个归一化参数进行学习优化
        if self.learned_norm:
            self.offset = torch.nn.Parameter(torch.tensor([self.args.initial_offset]).to(self.device), requires_grad=True)
            self.scaler = torch.nn.Parameter(torch.tensor([self.args.initial_scaler]).to(self.device), requires_grad=True)
            # 对参数设置不同的优化器参数
            self.optimzer.add_param_group({"params": [self.offset], 'lr': self.args.lr_offset })
            self.optimzer.add_param_group({"params": [self.scaler], 'lr': self.args.lr_scaler })
        else:
            self.offset = self.args.initial_offset
            self.scaler = self.args.initial_scaler
        # 学习率策略    
        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimzer, lr_lambda = lambda epoch: 1
            )
        else:
            self.lr_scheduler = lr_scheduler(self.optimzer)
        
        # 位姿更新学习
        if self.refine_poses and not self.skip_ckpt:
            print("待补充")
        else:
            self.pose_model = None
            self.pose_optimizer = None
            self.pose_lr_scheduler = None    
        
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.loss_dict = {
            "total loss": [],
            "pose gradient penalty": [],
            "pose coplanar penalty": [],
            "occ. bimodal penalty": [],
            "occ. mask penalty": [],
            "FFT reconstruction": [],
            "occupancy grounding penalty": [],
            "occupancy above penalty": []
        }
        self.checkpoints = []
        
        if self.args.mask: self.sin_epoch =0.0 if self.training else 1.0
        else: self.sin_epoch = None
        
        # 设置log路径
        # workspace/logs/log_radarfields.txt
        self.workspace.mkdir(exist_ok = True)
        self.log_path = self.workspace / "logs"
        self.log_path.mkdir(exist_ok = True)
        self.log_path = self.log_path / f"log_{self.name}.txt"
        self.log_ptr = open(self.log_path, "a+")
        # 设置checkpoint路径
        # checkpoint/
        self.ckpt_path = Path("checkpoints")
        self.ckpt_path.mkdir(exist_ok = True)
        # 设置模输出保存路径
        self.img_path = self.workspace / "imgs" / self.name
        self.img_path.mkdir(parents = True, exist_ok = True)
        # workspace/imgs/alpha_results
        self.alpha_grid_path = self.img_path / "alpha_results"
        self.alpha_grid_path.mkdir(exist_ok = True)
        # workspace/imgs/FFT
        self.FFT_path = self.img_path / "FFT"
        self.FFT_path.mkdir(exist_ok = True)
        
        # workspace/plots/radarfields
        self.plot_path = self.workspace / "plots" / self.name
        self.plot_path.mkdir(parents=True, exist_ok = True)
        # workspace/img/trajectories
        self.trajrctories_path = self.img_path / "trajectories"
        self.trajectories_path.mkdir(exist_ok = True)
        
        self.log(f'[INFO] Trainer: {self.name} | {self.device} | workspace: {self.workspace}')
        self.log(f'[INFO] # of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        self.log(f'[INFO] Writing outputs to workspace directory: {self.workspace}')
        
        # checkpoint
        if skip_ckpt: return
        # 从0开始
        if self.which_checkpoint == "scratch": self.log("[INFO] Training from scratch")
        # 模型+权重
        elif self.which_checkpoint == "latest":
            self.log("[INFO] LOading lastest checkpoint ...")
            self.load_checkpoint()
        # 仅模型
        elif self.which_checkpoint == "latest_model":
            self.log("[INFO] Loading checkpoint (model only) ... ")
            self.load_checkpoint(model_only = True)
        # 特定checkpoint
        else:
            self.log(f"[INFO] Loading {self.which_checkpoint} ...")
            self.load_checkpoint(self.which_checkpoint)

    def __del(self):
        if self.log_ptr is not None:
            self.log_ptr.close()
            
    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_ptr is not None:
            print(*args, file = self.log_ptr)
            self.log_ptr.flush() #关闭缓冲区 写入文件
        
    def load_checkpoint(self, checkpoint=None, model_only=False, demo=False):
        if checkpoint is None:
            checkpoint_list = sorted()