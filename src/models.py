import torch

from stable_baselines3 import PPO, SAC, TD3

##########################
#
# 可操作性を考慮したロス関数 
#
##########################

__all__ = ["ExtendModel"]

class ManipulabilityLoss:
    """SB3 の既存 loss に可操作性楕円の L_aux を足す"""
    def __init__(self, *args,
                 use_manip_loss: bool = False,
                 manip_coef: float = 0.01,  # 追加ロスの重み
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.manip_coef = manip_coef
        self.use_manip_loss = use_manip_loss
    
    # --- On-policy (PPO/A2C) ---
    def loss_actor_critic(          
        self, values, log_prob, entropy, advantages, returns
    ):
        # 既定ロスの 3 成分を取得
        policy_loss, value_loss, entropy_loss = super().loss_actor_critic(
            values, log_prob, entropy, advantages, returns
        )

        # 例：確率分布の分散を抑える補助ロス
        manip_loss = torch.mean(log_prob ** 2)
        if self.use_manip_loss:
            # ここでは例として “観測ベクトルの最後の d 次元” を
            # Jacobian 行列 J に見立て，det(JJᵀ) の √ を使う：
            #   w = √det(JJᵀ)  (Yoshikawa 1984)
            # 実際には env からフル Jacobian を取得する関数に置き換える
            obs_batch = self.rollout_buffer.observations.reshape(-1, self.observation_space.shape[0])
            J = obs_batch[:, -6:].reshape(-1, 2, 3)
            w = torch.sqrt(torch.det(torch.bmm(J, J.transpose(1,2))) + 1e-8)
            manip_loss = -torch.mean(w) # 大きいほど可操作性が高いため最小化する
            self.logger.record("train/manip_loss", manip_loss.detach().cpu().numpy())

        total_loss = (
            policy_loss
            + self.vf_coef * value_loss    # vf_coef：SB3で定義済み
            - self.ent_coef * entropy_loss # ent_coef：SB3で定義済み
            + self.manip_coef * manip_loss
        )
        return total_loss, policy_loss, value_loss, entropy_loss

    # --- Off-policy (SAC/TD3/DQN) ---
    def compute_loss(self, *args, **kwargs):            
        # SAC はタプル (actor_loss, critic_loss, ent_loss) を返す実装
        losses = super().compute_loss(*args, **kwargs)
        # リスト/タプルになっていても順序通り unpack 可
        actor_loss, critic_loss, *rest = losses
        
        if self.use_manip_loss:
            J = kwargs["data"]["observation"][:, -6].reshape(-1, 2, 3)
            w = torch.sqrt(torch.det(torch.bmm(J, J.transpose(1,2))) + 1e-8)
            manip_loss = -torch.mean(w)
            actor_loss = actor_loss + self.manip_coef * manip_loss
            self.logger.record("train/manip_loss", manip_loss.detach().cpu().numpy())
            
        return (actor_loss, critic_loss, *rest)
    

###############################
#
##


class ExtendModel:
    def __init__(self, 
                model_name :str ,
                policy: str,
                env,
                device : str,
                batch_size: int,
                use_manip_loss : bool = False,
                manip_coef: float = 0.1,
                policy_kwargs :dict = None,
                **kwargs,                
                ):
        self.model_name = model_name
        self.policy = policy
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.use_manip_loss = use_manip_loss
        self.manip_coef = manip_coef
        policy_kwargs = policy_kwargs or dict(net_arch=dict(pi=[256,256], vf=[256,256]))
        # ベースクラスを動的に決定 (Mixin + 元のアルゴリズム)
        if model_name == 'PPO':
            BaseCls = type('CustomPPO', (ManipulabilityLoss, PPO), {})
        elif model_name == 'SAC':
            BaseCls = type('CustomSAC', (ManipulabilityLoss, SAC), {})
        elif model_name == 'TD3':
            BaseCls = type('CustomTD3', (ManipulabilityLoss, TD3), {})
        else:
            raise ValueError(f"サポート外のモデルです: {model_name}")

        # エージェント生成
        """
        アルゴリズム特有の設定は個別に実施
        """
        algo_specific: dict[str, dict] = {
            "PPO": dict(n_steps=kwargs.pop("n_steps", 2048)),
            "SAC": dict(train_freq=kwargs.pop("train_freq", 1),
                        gradient_steps=kwargs.pop("gradient_steps", 1),
                        learning_starts=kwargs.pop("learning_starts", 10000)),
            "TD3": dict(train_freq=kwargs.pop("train_freq", 1),
                        gradient_steps=kwargs.pop("gradient_steps", 1),
                        learning_starts=kwargs.pop("learning_starts", 10000)),
        }
        self.agent = BaseCls(
            policy = self.policy,
            env = self.env,
            verbose=1,
            device=self.device,
            batch_size=self.batch_size,
            policy_kwargs=policy_kwargs,
            use_manip_loss=use_manip_loss,
            manip_coef=manip_coef,
            **algo_specific[model_name],
            **kwargs,
        )

        

    def learn(self, *args, **kwargs):
        return self.agent.learn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.agent, name)
    



    def loss_actor_critic(   # v2 系の例
        self,
        values, log_prob, entropy, advantages, returns
    ):
        # --- SB3 デフォルトの損失 ---
        policy_loss, value_loss, entropy_loss = super().loss_actor_critic(
            values, log_prob, entropy, advantages, returns
        )

        # --- 追加したい正則化や表現学習 loss など ---
        my_term = torch.mean(log_prob**2)      # 例：分散抑制項
        total_loss = (
            policy_loss
            + self.vf_coef * value_loss
            - self.ent_coef * entropy_loss
            + 0.01 * my_term                # ←自分で係数を決める
        )
        return total_loss, policy_loss, value_loss, entropy_loss


# model = PPOWithAuxLoss(
#    "MlpPolicy",
#    env,
#    vf_coef=0.5,
#    ent_coef=0.0,
#    verbose=1
#)
#model.learn(1_000_000)




###################################
# Class PPO(self, PPO):

