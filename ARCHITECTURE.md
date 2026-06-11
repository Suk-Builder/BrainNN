# BrainNN 架构设计

## Transformer 的本质问题

1. **Attention is soft selection** — 所有 token 都参与，没有真正的路由
2. **Autoregressive 生成** — 在所有可能里贪心选，不是思考
3. **Loss 是 token-level** — 不知道"为什么要说这个"
4. **没有时间维度** — KV cache 是显存模拟，不是神经机制

## 人脑的对应机制

- 丘脑 = 硬编码路由，不是 attention
- 海马体 = O(1)索引，不是 O(n²) attention  
- 基底核 = 动作选择，不是 autoregressive 生成
- 皮层柱 = 局部特征提取，Transformer 可以在这里

## 人脑 vs Transformer 的根本差异

### Transformer 的本质
- 输入 → tokenize → embedding → 一堆向量在同一个高维空间里
- 所有计算都是向量之间的运算
- 每个 token 的状态由位置决定，不是由细胞本身的特性决定

### 人脑的本质
- 每个神经元有自己的: 细胞类型、膜电位、空间位置、连接拓扑、时间常数
- 不是全局优化，是一群独立的振荡器互相抑制、互相激发
- **硬**的、**局域**的、**异步**的

## 训练数据

用高质量推理链而非海量语料:
- BSEM 推理链
- 数学证明过程
- 代码实现的思考过程
