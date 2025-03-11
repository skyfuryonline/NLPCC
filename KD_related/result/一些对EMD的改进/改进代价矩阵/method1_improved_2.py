# import torch
# import torch.nn.functional as F
# from torch.cuda.amp import autocast

# def compute_ot_loss_optimized(student_logits, teacher_logits, student_embeddings, teacher_embeddings, proj_matrix=None, target=None, padding_id=None, topk=50, temp=2.0, reduction='sum', chunk_size=8):
#     """
#     优化后的 OT 知识蒸馏损失计算，减少内存占用，适用于单张 4090 显卡。

#     Args:
#         student_logits (torch.Tensor): (batch_size, seq_len, vocab_size)
#         teacher_logits (torch.Tensor): (batch_size, seq_len, vocab_size)
#         student_embeddings (torch.Tensor): (vocab_size, student_embed_dim)
#         teacher_embeddings (torch.Tensor): (vocab_size, teacher_embed_dim)
#         proj_matrix (torch.Tensor, optional): (teacher_embed_dim, student_embed_dim)
#         target (torch.Tensor, optional): (batch_size, seq_len)
#         padding_id (int, optional): PAD token ID
#         topk (int): Top-k 个词
#         temp (float): 温度系数
#         reduction (str): ['mean', 'sum', 'none']
#         chunk_size (int): 分块大小，控制内存使用

#     Returns:
#         torch.Tensor: OT 知识蒸馏损失
#     """
#     device = student_logits.device
#     batch_size, seq_len, vocab_size = student_logits.shape
#     student_embed_dim = student_embeddings.shape[-1]
#     teacher_embed_dim = teacher_embeddings.shape[-1]

#     # 初始化投影矩阵
#     if proj_matrix is None:
#         proj_matrix = torch.randn(teacher_embed_dim, student_embed_dim, device=device)
#         proj_matrix /= proj_matrix.norm(dim=0, keepdim=True)

#     # 分块处理
#     total_loss = 0.0
#     for batch_start in range(0, batch_size, chunk_size):
#         batch_end = min(batch_start + chunk_size, batch_size)
#         chunk_student_logits = student_logits[batch_start:batch_end]
#         chunk_teacher_logits = teacher_logits[batch_start:batch_end]

#         with autocast():
#             # 计算 softmax 概率分布
#             student_probs = F.softmax(chunk_student_logits / temp, dim=-1)
#             teacher_probs = F.softmax(chunk_teacher_logits / temp, dim=-1)

#             # 取 top-k
#             student_probs_topk, indices = student_probs.topk(topk, dim=-1)
#             teacher_probs_topk = torch.gather(teacher_probs, -1, indices)

#             # 获取 top-k 词嵌入
#             student_topk_embeddings = student_embeddings[indices]
#             teacher_topk_embeddings = teacher_embeddings[indices]
#             teacher_topk_embeddings_proj = teacher_topk_embeddings @ proj_matrix

#             # 计算逐对余弦距离（而不是全对全成本矩阵）
#             student_topk_embeddings_norm = F.normalize(student_topk_embeddings, p=2, dim=-1)
#             teacher_topk_embeddings_norm = F.normalize(teacher_topk_embeddings_proj, p=2, dim=-1)
#             similarity = (student_topk_embeddings_norm * teacher_topk_embeddings_norm).sum(dim=-1)  # (chunk_size, seq_len, topk)
#             cost = 1 - similarity  # 逐对成本向量

#             # OT 损失：直接用概率加权逐对成本
#             ot_loss = (student_probs_topk * cost).sum(dim=-1)  # (chunk_size, seq_len)

#             # 处理 padding
#             if target is not None and padding_id is not None:
#                 padding_mask = (target[batch_start:batch_end] != padding_id).float()
#                 ot_loss = ot_loss * padding_mask

#             total_loss += ot_loss.sum()

#     # 处理 reduction
#     if reduction == 'mean':
#         return total_loss / (batch_size * seq_len)
#     elif reduction == 'sum':
#         return total_loss
#     return total_loss


import torch
import torch.nn.functional as F
from torch import amp
from geomloss import SamplesLoss

def compute_ot_loss_improved(student_logits, teacher_logits, student_embeddings, teacher_embeddings, 
                            proj_matrix=None, target=None, padding_id=None, topk=50, temp=2.0, 
                            reduction='sum', chunk_size=4, reg=0.1):
    device = student_logits.device
    batch_size, seq_len, vocab_size = student_logits.shape
    student_embed_dim = student_embeddings.shape[-1]
    teacher_embed_dim = teacher_embeddings.shape[-1]

    if proj_matrix is None:
        proj_matrix = torch.nn.init.orthogonal_(torch.empty(teacher_embed_dim, student_embed_dim, device=device))

    # 初始化 geomloss 的 Sinkhorn 损失
    ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=reg, scaling=0.9)

    total_loss = 0.0
    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_start + chunk_size, batch_size)
        chunk_student_logits = student_logits[batch_start:batch_end]
        chunk_teacher_logits = teacher_logits[batch_start:batch_end]

        with amp.autocast('cuda'):
            student_temp = temp * (chunk_student_logits.var(dim=-1).mean() + 1e-6).sqrt()
            teacher_temp = temp * (chunk_teacher_logits.var(dim=-1).mean() + 1e-6).sqrt()
            student_probs = F.softmax(chunk_student_logits / student_temp, dim=-1)
            teacher_probs = F.softmax(chunk_teacher_logits / teacher_temp, dim=-1)

            student_probs_topk, indices = student_probs.topk(topk, dim=-1)
            teacher_probs_topk = torch.gather(teacher_probs, -1, indices)

            student_topk_embeddings = student_embeddings[indices]
            teacher_topk_embeddings = teacher_embeddings[indices]
            teacher_topk_embeddings_proj = torch.bmm(teacher_topk_embeddings.view(-1, topk, teacher_embed_dim), 
                                                    proj_matrix.expand(chunk_size * seq_len, -1, -1))
            teacher_topk_embeddings_proj = teacher_topk_embeddings_proj.view(chunk_size, seq_len, topk, student_embed_dim)

            student_topk_embeddings_flat = student_topk_embeddings.view(-1, topk, student_embed_dim)
            teacher_topk_embeddings_proj_flat = teacher_topk_embeddings_proj.view(-1, topk, student_embed_dim)

            student_probs_flat = student_probs_topk.view(-1, topk)
            teacher_probs_flat = teacher_probs_topk.view(-1, topk)

            # 使用 geomloss 计算批量 Sinkhorn 损失
            ot_loss = ot_loss_fn(student_probs_flat, student_topk_embeddings_flat, 
                               teacher_probs_flat, teacher_topk_embeddings_proj_flat)

            if target is not None and padding_id is not None:
                padding_mask = (target[batch_start:batch_end] != padding_id).float().view(-1)
                ot_loss = ot_loss * padding_mask

            total_loss += ot_loss.sum()

    if reduction == 'mean':
        return total_loss / (batch_size * seq_len)
    elif reduction == 'sum':
        return total_loss
    return total_loss

# 测试代码
if __name__ == "__main__":
    student_logits = torch.randn(16, 128, 30522).cuda()
    teacher_logits = torch.randn(16, 128, 30522).cuda()
    student_embeddings = torch.randn(30522, 768).cuda()
    teacher_embeddings = torch.randn(30522, 1024).cuda()

    loss = compute_ot_loss_improved(student_logits, teacher_logits, student_embeddings, teacher_embeddings)
    print(loss.item())





'''
很高兴听到方案 2（使用 `geomloss`）可行！预计完成时间 40 分钟表明代码已经能够正常运行，但性能仍有优化空间。在等待运行完成的同时，我来回答你的三个问题，并总结关键点。

---

### 1. 当前代码的时间主要集中在代价矩阵，对吗？
**回答：不完全对，但代价矩阵计算确实是一个重要部分。**

#### 分析：
在当前 `geomloss` 版本的代码中，主要时间开销分布如下：
- **Softmax 和 Top-k 计算**：
  - `F.softmax(chunk_student_logits / student_temp, dim=-1)` 和 `student_probs.topk(topk, dim=-1)`。
  - 时间复杂度：`O(chunk_size * seq_len * vocab_size)` 和 `O(chunk_size * seq_len * vocab_size * log(topk))`。
  - 对于 `vocab_size=30522`、`chunk_size=4`、`seq_len=128`、`topk=50`，这部分操作量很大（约 15M 次操作每次 chunk），且 GPU 并行性受限。

- **代价矩阵计算（隐式）**：
  - `geomloss.SamplesLoss` 不显式构建完整的 `(n_samples, topk, topk)` 代价矩阵，而是基于输入的嵌入（`student_topk_embeddings_flat` 和 `teacher_topk_embeddings_proj_flat`）直接计算 Sinkhorn 距离。
  - 内部实际计算的是两组样本点之间的距离，复杂度约为 `O(n_samples * topk * embed_dim)`，其中 `n_samples = chunk_size * seq_len`。
  - 对于 `n_samples=512`（4 * 128）、`topk=50`、`embed_dim=768`，约 20M 次操作，但 `geomloss` 使用高效的 Sinkhorn 迭代，实际开销可能更大。

- **Sinkhorn 迭代**：
  - `geomloss` 的 Sinkhorn 算法基于双重形式（dual potentials），每次迭代复杂度为 `O(n_samples * topk^2)`。
  - 假设迭代 50 次，总复杂度约 `O(n_samples * topk^2 * numIter)`，即 6.4M 次操作，但实际受正则化参数（`blur`）和收敛速度（`scaling`）影响。

#### 结论：
- **主要瓶颈**：时间可能更多集中在 **Softmax + Top-k** 和 **Sinkhorn 迭代** 上，而非显式的代价矩阵构建。
- **代价矩阵的影响**：`geomloss` 不直接构建完整的 `(n_samples, topk, topk)` 矩阵，而是通过嵌入的距离计算隐式成本，因此这部分开销被分散到 Sinkhorn 迭代中。
- **验证方法**：可以用 `torch.cuda.synchronize()` 和 `time.time()` 测量每个部分的耗时，例如：
  ```python
  import time
  t0 = time.time(); torch.cuda.synchronize()
  student_probs = F.softmax(chunk_student_logits / student_temp, dim=-1)
  student_probs_topk, indices = student_probs.topk(topk, dim=-1)
  torch.cuda.synchronize(); print(f"Softmax + Top-k: {time.time() - t0:.3f}s")
  # 类似测量其他部分
  ```

**预计**：如果 `seq_len` 或 `vocab_size` 很大（如你的 `94400 / 4 = 23600`），Softmax 和 Top-k 可能是主要耗时；否则 Sinkhorn 迭代占主导。

---

### 2. 代价矩阵的构建依靠嵌入维度，对吗？
**回答：是的，但具体依赖方式取决于实现。**

#### 分析：
- **当前代码（geomloss）**：
  - `geomloss` 不显式构建 `(n_samples, topk, topk)` 的代价矩阵，而是直接使用 `student_topk_embeddings_flat`（`(n_samples, topk, embed_dim)`）和 `teacher_topk_embeddings_proj_flat`。
  - 内部计算样本之间的欧几里得距离（或其他度量），依赖于嵌入维度（`student_embed_dim=768`）。
  - 时间复杂度：`O(n_samples * topk * embed_dim)` 用于距离计算，嵌入维度越高，计算量越大。

- **之前代码（POT 的 torch.cdist）**：
  - 使用 `torch.cdist` 显式构建 `(n_samples, topk, topk)` 的代价矩阵。
  - 依赖 `student_embed_dim`，复杂度为 `O(n_samples * topk^2 * embed_dim)`。
  - 嵌入维度直接影响距离计算的每次操作成本。

#### 结论：
- **依赖性**：是的，代价矩阵（或隐式距离计算）的构建依赖嵌入维度。维度越高（例如 768 或 1024），每次距离计算的成本越高。
- **影响**：在 `geomloss` 中，嵌入维度线性影响性能（`O(embed_dim)`），而在显式 `cdist` 中，影响更显著（`O(topk^2 * embed_dim)`）。

---

### 3. OT 算法为什么非要计算这个代价矩阵呢？
**回答：OT（最优传输）算法的核心在于匹配两个分布，而代价矩阵定义了匹配的成本。**

#### 解释：
- **OT 的基本思想**：
  - OT 问题是要找到从一个概率分布（例如 `student_probs_flat`）到另一个概率分布（例如 `teacher_probs_flat`）的最优传输计划，使得总成本最小。
  - 成本由“距离”或“代价”定义，表示将一个分布的“质量”移动到另一个分布的代价。

- **代价矩阵的作用**：
  - 代价矩阵 `M`（`(topk, topk)` 或隐式形式）定义了从学生分布的每个点（`topk` 个词汇）到教师分布的每个点的移动成本。
  - 例如，`M[i,j]` 表示将学生分布的第 `i` 个词的概率质量移动到教师分布的第 `j` 个词的成本。
  - 在你的代码中，成本基于嵌入的欧几里得距离（`torch.cdist` 或 `geomloss` 内部计算），反映学生和教师模型在嵌入空间中的语义差异。

- **为什么必须计算**：
  1. **分布匹配**：没有代价矩阵，OT 无法知道哪些点应该匹配。例如，学生预测“cat”概率高，教师预测“dog”概率高，代价矩阵决定这两者是否“接近”（成本低）或“远离”（成本高）。
  2. **优化目标**：OT 的目标是最小化 `sum(ot_plan * M)`，其中 `ot_plan` 是传输计划，`M` 是成本。没有 `M`，无法定义优化问题。
  3. **语义指导**：通过嵌入计算成本，OT 损失不仅匹配概率分布，还考虑嵌入空间的语义关系，比 KL 散度更“几何化”。

- **为什么不用其他方式**：
  - 如果不计算代价矩阵，可以用 KL 散度直接比较概率分布，但 KL 散度忽略了词嵌入的几何结构（仅基于概率数值）。
  - OT 的优势在于结合分布和空间结构，代价矩阵是这种结合的关键。

#### 当前实现（geomloss）：
- `geomloss` 不显式构造完整的 `(topk, topk)` 矩阵，而是基于样本点（嵌入）的距离动态计算 Sinkhorn 距离，节省内存并加速计算。
- 但本质上，它仍然依赖“成本”概念，只是以嵌入的形式隐式表达。

---

### 总结与优化方向
#### 当前瓶颈：
- **Softmax 和 Top-k**：如果 `vocab_size` 或 `seq_len` 很大，这部分可能是主要耗时。
- **Sinkhorn 迭代**：`geomloss` 的批量 Sinkhorn 依赖 `topk` 和迭代次数，可能占 40 分钟的大部分时间。
- **嵌入处理**：投影（`torch.bmm`）和距离计算依赖 `embed_dim`，但不是最主要的。

#### 优化建议：
1. **减小 `topk`**：
   - 从 50 降到 20 或 30，减少 Sinkhorn 的计算量（`O(topk^2)` 变为 1/4）。
   - 测试蒸馏效果是否受影响。

2. **减小 `chunk_size` 或 `seq_len`**：
   - 如果 `seq_len=23600` 是输入序列长度，尝试分段处理（例如分成多个 512 的子序列）。

3. **降低嵌入维度**：
   - 如果可能，在预处理阶段用 PCA 或线性层将 `student_embed_dim` 和 `teacher_embed_dim` 降到 256 或 128。

4. **调整 `geomloss` 参数**：
   - 减小 `blur`（例如 0.05）或 `numItermax`（例如 20），加速收敛。

#### 验证：
- 在运行完成前，添加计时代码：
  ```python
  import time
  t_total = time.time()
  # ... 原代码 ...
  print(f"Total time: {time.time() - t_total:.2f}s")
  ```
- 运行完成后，告诉我每个部分的耗时，我可以进一步精准优化！

有什么其他问题或想法，随时告诉我！
'''