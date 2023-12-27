import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import time
import torch.nn as nn


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MarioNet, self).__init__()

        c, h, w = input_dim
        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)


# 加载环境
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 定义预训练模型的路径
model_path = 'saved_models/mario_net_29.chkpt'

# 创建模型实例
model = MarioNet(input_dim=(4, 84, 84), output_dim=7)  # 根据您的模型定义，确保输入和输出维度正确

# 加载预训练模型的状态字典
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()  # 设置为评估模式

done = True
total_reward = 0
start_time = time.time()

while True:
    if done:
        state = env.reset()

    # 对状态进行必要的预处理
    processed_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # 使用模型预测动作
    with torch.no_grad():
        action = model(processed_state).argmax().item()

    next_state, reward, done, trunc, info = env.step(action)
    total_reward += reward

    # 渲染环境
    env.render()

    # 检查是否通关
    if info['flag_get']:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"马里奥成功通关！总奖励: {total_reward}, 耗时: {elapsed_time} 秒")
        break

env.close()
