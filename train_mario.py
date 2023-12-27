import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True,render_mode='human')
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human',apply_api_compatibility=True)

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """每隔 `skip` 帧返回一次"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """重复动作并累计奖励"""
        total_reward = 0.0
        for i in range(self._skip):
            # 累计奖励并重复相同的动作
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # 重排 [H, W, C] 数组为 [C, H, W] 张量
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# 将包装器应用于环境
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)


class Mario:
    def __init__(self):
        pass

    def act(self, state):
        """给定一个状态，选择一个ε-贪婪（epsilon-greedy）动作"""
        pass

    def cache(self, experience):
        """将经验添加到内存中"""
        pass

    def recall(self):
        """从内存中抽样经验"""
        pass

    def learn(self):
        """使用一批经验更新在线动作值（Q值）函数"""
        pass


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        # 初始化玛丽奥的状态维度、动作维度和保存目录
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # 检测设备是否支持CUDA，如果支持，则使用CUDA，否则使用CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化Mario的DNN以预测最优动作 - 我们在后面的“学习”部分实现这一部分
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        # 设置探索率相关的参数
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # 每隔多少经验保存一次Mario Net
        self.save_every = 5e5

    def act(self, state):
        """
        给定一个状态，选择一个ε-贪婪动作并更新步骤的值。

        输入:
        state(``LazyFrame``): 当前状态的单个观察值，维度为 (state_dim)

        输出:
        ``action_idx`` (``int``): 表示玛丽奥将执行的动作的整数索引
        """
        # 探索
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # 利用
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # 减少探索率
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # 增加步骤
        self.curr_step += 1
        return action_idx



class Mario(Mario):  # 继承父类以保持连续性
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        # 初始化记忆回放缓冲区
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32  # 批处理大小

    def cache(self, state, next_state, action, reward, done):
        """
        将经验存储到self.memory（回放缓冲区）中

        输入:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        # 辅助函数，如果输入是元组，则取第一个元素
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        # 将LazyFrame转换为数组
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        # 转换为PyTorch张量
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # 将经验添加到记忆中
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        从记忆中检索一批经验
        """
        # 从记忆中随机抽样一批经验，并将其移到适当的设备上
        batch = self.memory.sample(self.batch_size).to(self.device)

        # 获取批量中的各个经验项
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


class MarioNet(nn.Module):
    """迷你CNN结构
  输入 -> (conv2d + relu) x 3 -> 展平 -> (dense + relu) x 2 -> 输出
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"期望输入的高度为: 84, 实际为: {h}")
        if w != 84:
            raise ValueError(f"期望输入的宽度为: 84, 实际为: {w}")

        # 构建在线模型和目标模型
        self.online = self.__build_cnn(c, output_dim)
        self.target = self.__build_cnn(c, output_dim)

        # 将在线模型的权重加载到目标模型中
        self.target.load_state_dict(self.online.state_dict())

        # 冻结目标模型的参数，不进行反向传播更新
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """前向传播函数，根据模型参数决定是在线模型还是目标模型"""
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        """构建CNN结构"""
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9  # 定义折扣因子为 0.9

    def td_estimate(self, state, action):
        """
        计算TD估计值
        """
        # 获取当前动作对应的在线模型下的Q值
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        计算TD目标值
        """
        # 获取下一个状态在在线模型下的Q值
        next_state_Q = self.net(next_state, model="online")

        # 选择在下一个状态中最优的动作
        best_action = torch.argmax(next_state_Q, axis=1)

        # 获取目标模型下的Q值
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]

        # 计算TD目标值
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())



class Mario(Mario):
    def save(self):
        """
        保存 MarioNet 模型和探索率
        """
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")



def load_model_weights(agent, path):
    """加载模型权重到代理网络"""
    if torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    agent.net.load_state_dict(state_dict['model'])
    agent.exploration_rate = state_dict['exploration_rate']



# 定义一个新的 Mario 类，该类从基础 Mario 类继承
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)

        # 设置开始训练前需要的最小经验数量
        self.burnin = 1e4

        # 每3个经验更新一次 Q_online
        self.learn_every = 3

        # 每1e4个经验，进行 Q_target 和 Q_online 的同步
        self.sync_every = 1e4

    # 定义学习方法
    def learn(self):
        # 如果当前步数可以被 sync_every 整除，同步 Q_target 和 Q_online
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # 如果当前步数可以被 save_every 整除，保存模型
        if self.curr_step % self.save_every == 0:
            self.save()

        # 如果当前步数小于 burnin，暂时不进行学习
        if self.curr_step < self.burnin:
            return None, None

        # 如果当前步数不能被 learn_every 整除，暂时不进行学习
        if self.curr_step % self.learn_every != 0:
            return None, None

        # 从记忆中随机抽样获取经验数据
        state, next_state, action, reward, done = self.recall()

        # 计算 TD 估计值
        td_est = self.td_estimate(state, action)

        # 计算 TD 目标值
        td_tgt = self.td_target(reward, next_state, done)

        # 通过 Q_online 反向传播并更新网络权重
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


# 导入所需库
import numpy as np
import time, datetime
import matplotlib.pyplot as plt

# 定义 MetricLogger 类用于日志记录和性能度量
class MetricLogger:
    def __init__(self, save_dir):
        # 设置保存日志文件路径
        self.save_log = save_dir / "log"
        # 在日志文件中写入列标题
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        # 设置保存图像的路径
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # 初始化历史性能指标
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # 初始化移动平均指标
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # 初始化当前的回合指标
        self.init_episode()

        # 记录时间
        self.record_time = time.time()

    # 记录每一步的信息
    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    # 标记回合结束
    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    # 初始化当前回合的指标
    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    # 记录每一回合的性能指标
    def record(self, episode, epsilon, step):
        # 计算最近100回合的平均性能
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        # 计算时间差
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        # 打印当前回合的性能指标
        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        # 在日志文件中追加当前回合的性能指标
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        # 绘制并保存性能图像
        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))


import keyboard

pk_flag = True
import threading
import os

def rm_temp():
    print("删除临时文件！")
    subprocess.run(["python","rm_temp.py"])

import subprocess

def run_main_py():
    print("启动线程！！！！！")
    subprocess.run(["python", "main.py"])

rm_thread = threading.Thread(target=rm_temp)
rm_thread.start()

if __name__ == "__main__":

    # 等待用户按下Enter键开始训练
    if pk_flag:
        main_thread = threading.Thread(target=run_main_py)
        main_thread.start()


    print("按下Enter键开始训练...")
    keyboard.wait("enter")

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    # 是否加载预训练模型的标志
    load_pretrained_model = True
    pretrained_model_path = "./saved_models/mario_net_29.chkpt"

    if load_pretrained_model:
        if pretrained_model_path:
            print(f"找到预训练模型：{pretrained_model_path}")
            load_model_weights(mario, pretrained_model_path)
            print("已加载预训练模型")
        else:
            print("未找到任何.pth文件")

    episodes = 40000

    for e in range(episodes):
        state = env.reset()
        while True:
            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            logger.log_step(reward, loss, q)
            state = next_state
            env.render()
            time.sleep(0.185)
            if done or info["flag_get"]:
                break
        logger.log_episode()
        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

