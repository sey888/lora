import torch
import numpy as np
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import cv2
import os

from depth_anything_v2_metric.depth_anything_v2.dpt import DepthAnythingV2
from .utils import LoRA_Depth_Anything_v2

from argparse import Namespace
from .models import register
from depth_anything_utils import Resize, NormalizeImage, PrepareForNet

class PanDA(nn.Module):
    def __init__(self, args):
        """
        PanDA model for depth estimation
        """
        super().__init__()
        
        midas_model_type = args.midas_model_type
        fine_tune_type = args.fine_tune_type
        min_depth = args.min_depth
        self.max_depth = args.max_depth
        lora = args.lora
        train_decoder = args.train_decoder
        lora_ranks = args.lora_ranks

        # Pre-defined setting of the model
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Load the pretrained model of depth anything
        depth_anything = DepthAnythingV2(**{**model_configs[midas_model_type], 'max_depth': 1.0})
        
        # 1. 基础权重加载
        if fine_tune_type == 'none':
            depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{midas_model_type}.pth'), strict=False)
        elif fine_tune_type == 'hypersim':
            depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_hypersim_{midas_model_type}.pth'), strict=False)
        elif fine_tune_type == 'vkitti':
            depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_vkitti_{midas_model_type}.pth'), strict=False)
        elif fine_tune_type == "backbone":
            depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{midas_model_type}.pth'), strict=False)
        
        # 2. 自动检测并加载阶段一收敛权重 (Best Model)
        stage1_path = 'tmp/_train/best/model.pth'
        is_stage2 = False
        
        if lora:
            self.core = depth_anything
            LoRA_Depth_Anything_v2(depth_anything, lora_ranks=lora_ranks)
            
            if os.path.exists(stage1_path):
                print(f">>> [Stage 2] 检测到阶段一权重: {stage1_path}")
                print(">>> 正在加载阶段一权重并冻结 LoRA 参数...")
                self.load_state_dict(torch.load(stage1_path), strict=False)
                is_stage2 = True
                
                # 冻结所有参数，只开放 PolarAttention 的梯度
                for name, param in self.named_parameters():
                    if "polar_attention" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print(">>> [Stage 2] 冻结完成，仅 PolarAttention 模块可训练。")
            
            if not train_decoder and not is_stage2:
                for param in self.core.depth_head.parameters():
                    param.requires_grad = False
        else:
            self.core = depth_anything
    
    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Forward of erp image
        erp_pred = self.core(image)
        erp_pred = erp_pred.unsqueeze(1)
      
        outputs = {}
        outputs["pred_depth"] = erp_pred * self.max_depth

        return outputs
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)["pred_depth"]
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size * 2,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
    
@register('panda')
def make_model(midas_model_type='vits', fine_tune_type='none', min_depth=0.1, max_depth=10.0, lora=True, train_decoder=True, lora_ranks=None):
    args = Namespace()
    args.midas_model_type = midas_model_type
    args.fine_tune_type = fine_tune_type
    args.min_depth = min_depth
    args.max_depth = max_depth
    args.lora = lora
    args.train_decoder = train_decoder
    args.lora_ranks = lora_ranks
    return PanDA(args)
