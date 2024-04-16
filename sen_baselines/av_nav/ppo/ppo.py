import torch
import torch.nn as nn
import torch.optim as optim

from ss_baselines.av_nav.ppo.policy import Policy
from ss_baselines.common.rollout_storage import RolloutStorage

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self, 
        actor_critic: Policy, 
        clip_param, 
        ppo_epoch, 
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
        use_normalized_advantage=True) -> None:
        
        super().__init__()
        
        self.actor_critic = actor_critic
        
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        # Adam优化器, 用于优化actor_critic中的所有可训练参数. lr是学习率, eps是一个小常数, 用于防止分母为0
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        
    # PPO算法是一个策略类, 而不是网络, 因此不实现forward算法
    def forward(self, *x):
        raise NotImplementedError
    
    # 获得优势函数. 优势函数用于评估智能体在给定状态下执行某个动作相对于平均表现的优势
    # A(s, a) = Q(s, a) - V(s), 
    # A表示状态s下采取动作a的优势值
    # Q表示状态s下采取动作a的State-Action函数值
    # V表示状态s下的价值函数
    # 价值函数V在Policy中的实现应该是CriticHead
    # 动作函数Q在RolloutStorage中有实现, compute_returns(), returns = gae + value_preds
    # 这里获取的优势值就是GAE
    def get_advantage(self, rollouts: RolloutStorage):
        # 二者的维度都是 (num_steps + 1, num_envs, 1), 返回值的维度是(num_steps, num_envs, 1)
        advantage = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantage
        return (advantage - advantage.mean()) / (advantage.std() + EPS_PPO)
    
    # 使用num_steps个步骤的rollout信息更新网络
    # 计算动作损失, 值损失和熵, 并使用优化器optimizer更新actor-critic网络
    def update(self, rollouts: RolloutStorage):
        # print("***************************")
        # print("PPO UPDATE")
        
        advantages = self.get_advantage(rollouts)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        # PPO epoch = 4, 在yaml.PPO中定义的
        # 在执行完num_steps个动作之后, 长序列更新重复进行4次
        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )
            
            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                
                # Reshape to do in a single forward pass for all steps
                # 将内容输入Pocily网络, 获得的结果是value, action_log_probs, distribution_entropy, rnn_hidden_states
                # 其中value是V值, State-Value. 这里输入的是batch size = num_steps的批处理向量, value和action_log_probs
                # 应该是包含batch size的向量, dist_entropy应该是一个平均值, 应该是一个数
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )
                
                # 使用e^x计算新策略和旧策略之间动作概率的比率. ratio用于计算PPO算法中的surrogate函数
                # 是PPO算法的核心, 用于平衡策略更新的幅度. 通过使用ratio和clip参数, 可以限制策略更新的幅度
                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                # 不带剪裁的原始surrogate函数.
                surr1 = ratio * adv_targ
                # 带剪裁的surrogate函数. 这里将ratio限制在一个范围内, 这个范围是
                # [1.0 - self.clip_param, 1.0 + self.clip_param]. torch.clamp用于将
                # ratio限制在这个范围内. 
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    ) * adv_targ
                )
                # 取二者中的较小的作为目标函数来最大化其值, 从而实现策略的优化
                # torch.min计算了两个surrogate的逐元素最小值. 而mean()计算了这些最小值的平均值
                # 最后, 为了将问题转化为最小化损失问题, 取平均值的负值. 通过最小化action_loss, 可以更新策略
                action_loss = -torch.min(surr1, surr2).mean()
                
                # 计算截断值函数损失. 
                if self.use_clipped_value_loss:
                    # 计算预测目标值, 用于修正返回的目标值. 
                    # 后面的减法是将预测值与目标值的差值限定在给定范围内. 
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    # 计算未截断的值损失
                    value_losses = (values - return_batch).pow(2)
                    # 计算截断的值损失
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    # 值损失取二者逐元素最大值的平均值, 再*0.5
                    value_loss = (0.5 * torch.max(value_losses, value_losses_clipped).mean())
                else:
                    # 直接取目标值与预测值的平方差的一半, 再取平均
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    
                # print("***************************")
                # print("PPO optimizer")
                
                # 对模型进行训练和优化. 重置模型参数的梯度, 为了防止重复计算, 每次迭代都需要显示的将梯度清零
                self.optimizer.zero_grad()
                # 计算总损失total_loss, 是值函数损失, 动作函数损失和分布熵损失的甲醛. 
                total_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)
                
                self.before_backward(total_loss)
                # 调用反向传播, 计算损失相对于模型参数的梯度
                total_loss.backward()
                self.after_backward(total_loss)
                
                # 梯度裁减
                self.before_step()
                # 对模型参数进行更新, 使用之前计算出的梯度
                self.optimizer.step()
                self.after_step()
                
                # 将损失累加到相应的损失变量, 以便在一个时期内进行统计
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                
        # 计算三个参数的平均值
        num_updates = self.ppo_epoch * self.num_mini_batch
        
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
        
    def before_backward(self, loss):
        pass
    
    def after_backward(self, loss):
        pass
    
    # 梯度裁减
    def before_step(self):
        # 将actor-critic网络的参数的梯度限制在一个最大值 max_grad_norm 之下
        nn.utils.clip_grad.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )
        
    def after_step(self):
        pass
