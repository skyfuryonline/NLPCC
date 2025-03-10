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

'''
温度𝑇影响 softmax 之后的概率分布，确实会导致新的分布与原来的不同。然而，当温度升高时，KL 散度通常会变小，这背后的核心原因是：
高温𝑇>1让 softmax 分布变得更加平滑，使得它与基准分布的熵差距缩小，从而减少 KL 散度。

(1) 低温导致 KL 散度增大
如果T<1，softmax 会变得更加尖锐，使得概率分布偏离基准分布，例如：
T=0.5：P(i)变得更接近 one-hot，这意味着某些类别的概率急剧下降到接近 0，导致 KL 散度更大。
因为 KL 散度包含logP/Q，当Q(i)变得极端（比如趋近 0）时，会导致 KL 值大幅增长。

(2) 高温让 KL 散度变小
如果T>1，softmax 变得更平滑：
P(i)和 Q(i)的分布更加接近，意味着logP/Q 的值会更小。
极端概率值减少，KL 散度计算时不会出现特别大的贡献项。

温度影响 softmax：
低温 T<1T < 1T<1 → 概率分布更极端化 → KL 散度增大。
高温 T>1T > 1T>1 → 概率分布更平滑 → KL 散度变小。
KL 散度与信息熵的关系：
低温 让 softmax 分布变得更“确定”（熵小），导致 KL 散度增加。
高温 让 softmax 分布变得更“模糊”（熵大），KL 散度减少。

一个小细节：KL散度需要2个分布，所以：
假设我们以 P1（T=1时的 softmax）为基准；


'''