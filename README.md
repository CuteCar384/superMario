#### 超级马里奥人机大战
1、创建一个新的环境
> conda create -n mario python=3.10.12 -y

2、激活环境
> conda activate mario

3、永久设置pip清华镜像
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

4、安装依赖
> pip install -r requirements.txt

5、运行train_mario.py，将打开两个窗口，小窗口是机器人游戏，大窗口是您的游戏
其中，Enter进入游戏，左右箭头控制方向，A键跳跃，S键加速，D键发射子弹。
