import matplotlib
matplotlib.use('TkAgg')  # 更换后端
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.widgets import Slider

# 设置原始 logits
logits = [2.0, 1.0, 0.1]

def softmax_with_temperature(logits, t):
    """计算具有温度 t 的 softmax 分布"""
    logits = torch.tensor(logits, dtype=torch.float32)
    return F.softmax(logits / t, dim=-1).numpy()

def kl_divergence(p, q):
    """计算 KL 散度 D_KL(P || Q)"""
    p, q = torch.tensor(p, dtype=torch.float32), torch.tensor(q, dtype=torch.float32)
    return F.kl_div(q.log(), p, reduction='sum').item()

# 初始温度
t_init = 1.0

def update(val):
    t = slider.val
    q = softmax_with_temperature(logits, t)
    kl_value = kl_divergence(softmax_with_temperature(logits, t_init), q)
    line.set_ydata([kl_value])
    line.set_xdata([t])
    fig.canvas.draw_idle()

# 创建图形
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_xlabel('Temperature (t)')
ax.set_ylabel('KL Divergence w.r.t. t=1.0')
ax.set_title('Effect of Temperature on KL Divergence')
ax.set_xlim(0.1, 5.0)
ax.set_ylim(0, 0.2)
ax.grid()

# 初始点
q_init = softmax_with_temperature(logits, t_init)
kl_init = kl_divergence(q_init, q_init)
line, = ax.plot([t_init], [kl_init], 'bo', markersize=8)

# 添加滑动条
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Temperature', 0.1, 5.0, valinit=t_init)
slider.on_changed(update)

plt.show()