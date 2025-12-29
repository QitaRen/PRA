import torch
import torch.nn as nn
import torch.nn.functional as F
# [修改点 1] 使用 DDIM 而不是 DDPM
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

from src.conditional_unet1d import ConditionalUnet1D
from src.normalizer import LinearNormalizer

class DiffusionPolicy(nn.Module):
    def __init__(self, 
                 action_dim, 
                 obs_dim, 
                 # [修改点 2] Config 中的 horizon 是 8
                 pred_horizon=8, 
                 num_inference_steps=10, # Config 中是 10
                 # [修改点 3] 依据 Config 调整网络架构参数
                 down_dims=(512, 1024, 2048),
                 diffusion_step_embed_dim=128,
                 kernel_size=5,
                 n_groups=8,
                 condition_type="film",
                 ):
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.num_inference_steps = num_inference_steps
        
        self.normalizer = LinearNormalizer()

        # [修改点 4] 依据 Config 调整 Scheduler
        # prediction_type='sample' 意味着模型预测的是 x0 (原始动作) 而不是 noise
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100, # Config: 100
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='sample' 
        )

        # [修改点 5] global_cond_dim 必须匹配 Teacher 的期望 (320 * 5 = 1600)
        # 我们在 Agent forward 里做 repeat，但这里定义模型时要写 1600
        teacher_global_cond_dim = obs_dim * 5 

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=teacher_global_cond_dim, # 1600
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True
        )

    def fit_normalizer(self, dataloader):
        print("Fitting normalizer with student data...")
        all_actions = []
        for batch in tqdm(dataloader, desc="Collecting actions"):
            actions = batch['action']
            all_actions.append(actions)
        all_actions = torch.cat(all_actions, dim=0)
        self.normalizer.fit(data={'action': all_actions}, last_n_dims=1, mode='limits')
        print("✅ Normalizer fit complete.")

    def compute_loss(self, action, global_cond):
        """
        Args:
            action: (B, T, D) GT Action
            global_cond: (B, 1600) Expanded Context
        """
        # 1. 归一化
        n_action = self.normalizer.normalize({'action': action})['action']
        B = n_action.shape[0]
        
        # 2. 采样噪声
        noise = torch.randn_like(n_action)
        
        # 3. 采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=n_action.device
        ).long()
        
        # 4. 加噪
        noisy_trajectory = self.noise_scheduler.add_noise(
            original_samples=n_action, 
            noise=noise, 
            timesteps=timesteps
        )
        
        # 5. 模型预测
        # 注意：Teacher 训练时预测的是 'sample' (原始动作)，不是 noise！
        pred = self.model(
            sample=noisy_trajectory, 
            timestep=timesteps, 
            global_cond=global_cond
        )
        
        # 6. 计算 Loss
        # prediction_type='sample' -> Target 是 n_action (原始动作)
        # prediction_type='epsilon' -> Target 是 noise
        if self.noise_scheduler.config.prediction_type == 'sample':
            target = n_action
        elif self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        else:
            raise ValueError(f"Unsupported prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(pred, target)
        return {'loss': loss}

    @torch.no_grad()
    def predict_action(self, global_cond):
        B = global_cond.shape[0]
        device = global_cond.device
        
        # DDIM 初始噪声
        shape = (B, self.pred_horizon, self.action_dim)
        noisy_trajectory = torch.randn(shape, device=device)
        
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            model_output = self.model(
                sample=noisy_trajectory, 
                timestep=t, 
                global_cond=global_cond
            )
            # DDIM Step
            noisy_trajectory = self.noise_scheduler.step(
                model_output, t, noisy_trajectory
            ).prev_sample
            
        action = self.normalizer.unnormalize({'action': noisy_trajectory})['action']
        return {'action_pred': action}