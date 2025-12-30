# M-GAN: Multi-Modal Generative Adversarial Network for Cancer Gene Identification

## 简介

M-GAN是一个基于多模态生成对抗网络的癌症基因识别框架。该模型通过整合多视图特征数据，使用生成对抗网络和图神经网络来识别潜在的癌症基因。模型采用共享生成器和特异性生成器的架构，结合注意力机制来融合不同模态的特征表示。

## 核心特性

- **多模态特征处理**: 支持按16维划分的多通道特征输入
- **生成对抗网络**: 实现共享生成器和特异性生成器的对抗训练
- **图神经网络**: 基于ChebConv的图编码器处理网络拓扑结构
- **注意力机制**: 自动学习不同模态的重要性权重
- **集成学习**: 包含多种预测模型的动态集成方法

## 模型架构

### 主要组件

1. **GNNEncoder**: 图神经网络编码器
   - 使用ChebConv进行图卷积操作
   - 包含两层图卷积层，每层后接ReLU和Dropout
   - 支持边dropout增强鲁棒性

2. **Generator**: 生成器网络
   - 共享生成器和特异性生成器
   - 集成注意力权重机制
   - 为每个模态学习256维嵌入表示

3. **Discriminator**: 判别器网络
   - 包含模态间交互权重矩阵
   - 计算所有模态对之间的交互特征
   - 使用双线性变换提取交互信息

4. **Predictor**: 预测网络
   - 基于图编码器和多层感知机
   - 输出癌症基因预测概率

### 训练策略

- **对抗训练**: 生成器和判别器交替优化
- **多任务损失**: 结合GAN损失和预测损失
- **动态权重**: 通过softmax归一化注意力权重
- **学习率调度**: 支持ReduceLROnPlateau调度器

## 使用方法

### 命令行参数

```bash
python M-GAN.py --dev cuda --epochs 500
```

**参数说明**:
- `--dev`: 设备选择 (cpu, cuda, cuda:0, cuda:1, cuda:2, cuda:3)
- `--epochs`: 训练轮数 (可选: 1, 2, 10, 20, 50, 100, 500, 1000, 1500, 2500, 3000)

### 数据要求

模型期望以下数据文件在`./data/`目录下：
- `CPDB_processed.pkl`: 预处理后的特征数据
- `k_sets.pkl`: 交叉验证设置数据

### 特征格式

- 原始特征矩阵应按16维进行通道划分
- 支持任意数量的特征通道
- 每个节点包含图结构信息和特征向量

## 文件结构

```
M-GAN.py
├── load_datasets()        # 数据加载和预处理
├── GNNEncoder           # 图神经网络编码器
├── Generator            # 生成器网络
├── Discriminator        # 判别器网络
├── Predictor           # 预测网络
├── Train              # 训练类
├── Experiment         # 实验评估类
└── 其他增强组件...
```

## 核心功能

### 数据加载 (load_datasets)

```python
def load_datasets(args):
    # 加载CPDB_processed.pkl
    # 构建KNN图 (k=10)
    # 生成合成标签
    # 创建训练掩码
    # 返回处理后的数据
```

### 特征通道划分

```python
# 将特征按每16维划分为一个通道
features = [x[:, i:i+16] for i in range(0, x.shape[1], 16)]
args.n_modality = len(features)  # 自动确定模态数量
```

### 训练流程

1. **生成器训练阶段**: 
   - 前向传播生成特征
   - 计算GAN损失和预测损失
   - 反向传播更新参数

2. **判别器训练阶段**:
   - 判别真实和生成的特征
   - 更新判别器参数

3. **评估阶段**:
   - 计算AUC和AUPR指标
   - 保存训练结果

## 输出结果

训练完成后，结果保存在`./output/`目录下，包含：
- 训练集和测试集的AUC/AUPR曲线数据
- 详细的性能指标记录
- 每轮训练的可视化结果

## 依赖环境

```
PyTorch >= 1.13.1
torch_geometric >= 2.3.1
scikit-learn >= 0.22
numpy >= 1.21.6
pandas >= 1.1.5
tensorflow >= 2.x
imbalanced-learn >= 0.8.0
matplotlib >= 3.5.0
networkx >= 2.6.0
```

## 高级功能

### 增强组件

1. **ContrastiveLearningEnsemble**: 对比学习集成
2. **EnhancedAdaptiveFeature**: 增强自适应特征
3. **MultiScaleGraphAttention**: 多尺度图注意力
4. **DynamicGraphLearning**: 动态图学习
5. **GraphStructureAdaptiveEnhancement**: 图结构自适应增强
6. **SelfSupervisedPretraining**: 自监督预训练

### 评估指标

- **AUC**: ROC曲线下面积
- **AUPR**: 精确率-召回率曲线下面积
- **混淆矩阵**: 详细的分类结果分析

## 注意事项

1. **内存要求**: 大型数据集可能需要GPU支持
2. **训练时间**: 根据数据集大小和训练轮数调整
3. **参数调优**: 可根据具体任务调整学习率、权重衰减等超参数
4. **数据预处理**: 确保输入数据格式符合要求

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少批大小或使用CPU训练
2. **训练不稳定**: 调整学习率或增加正则化
3. **收敛缓慢**: 启用学习率调度器

### 调试模式

模型内置了详细的训练日志输出，可通过设置`verbose=True`启用详细模式。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 引用

如果使用本代码，请引用相关论文并注明M-GAN框架的使用。

---

**作者**: M-GAN开发团队  
**版本**: 1.0.0  
**最后更新**: 2025年