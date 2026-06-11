# BrainNN · 仿脑神经网络

**核心思路**: 不是靠 Scale堆参数，是靠架构模仿人脑的稀疏激活和信息路由机制。

## 架构

```
输入文本 → 丘脑路由(硬编码) → 皮层柱专家 → 基底核选择 → 海马体索引 → 输出
```

Transformer 只在皮层柱内部做特征提取，不再是全局推理引擎。

## 核心组件

### 丘脑路由 (Thalamus Router) — 已实现 v5
- 零参数，硬编码路由规则
- 问题类型 → 分配专家皮层柱（逻辑/类比/知识/代码/创意）
- 不是 attention soft selection，是 if-else hard routing

### 海马体记忆 (Hippocampus Memory) — 待做
- 网格编码把概念映射到高维空间
- 余弦相似度 O(1) 检索

### 基底核选择 (Basal Ganglia) — 待做
- 每个皮层柱生成候选输出，评分函数选出最优

## 当前版本

- v5: Qwen2.5-14B-Instruct + Zoe 14B-v6 LoRA (VRAM 30.1GB)
- v4: Qwen2.5-3B-Instruct + Zoe 3B-v9 LoRA (VRAM 12.5GB)

## 运行

```python
POST http://127.0.0.1:7861/api/chat
{"message": "你的问题", "temperature": 0.6, "max_tokens": 400}
```

## 为什么还要改 Transformer

Transformer 是概率分布拟合器，不是因果推理器。
它不是在"想"，它只是在吐统计学上最可能的下一个token。

下一步: Transformer退化为皮层柱内部特征提取器，推理引擎换成丘脑+基底核。
