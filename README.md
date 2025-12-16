# PCMA

PCMA (Paired Carrier Multiple Access) 信号分离项目

## 模型训练部分

- **model_complex.py** - 定义SignalSeparator模型架构，包含编码器、解码器和分离网络
- **generate_sim_dataset.py** - 生成仿真数据集，创建混合信号用于模型训练
- **torch_signal_sim.py** - 纯PyTorch版本的信号生成模块，支持可微分的参数，用于训练过程中的数据生成
- **train_SignalSeparator.py** - 模型训练主程序，支持分布式训练
- **train_SignalSeparator_v2.py** - 改进版训练脚本，添加了噪声一致性损失和EVM损失
- **test_sim_SignalSeparator.py** - 测试仿真模型性能，评估SER、EVM等指标
- **utils.py** - 工具函数，包含符号聚类、SER计算等辅助功能
- **example.py** - 最小示例代码，演示完整的信号生成、模型推理、SER评估和信号重建流程
- **viz_metrics.py** - 可视化评估指标（SER、EVM等），生成统计图表


## 数据分析部分

- **compensation.py** - Costas环补偿合理性分析，用于相位补偿
- **estimate_h.py** - 估计成型滤波器（RRC滤波器）参数
- **split_data.py** - 数据裁剪和分割工具
- **utils_compensation.py** - 补偿相关的工具函数

## 数据路径

- 采集数据的原始数据存放在 `/nas/datasets/LYX/PCMA` 下，关注"QPSK_16"、"8PSK_16"以及"16QAM_16"，其中QPSK的SPS为8，其他两种为16，成型滤波器为RRC(beta=0.33)，具体保存格式为I、Q、I、Q…… 每个数据点为一个short。
- 仿真生成的8PSK数据放在 `/nas/datasets/yixin/PCMA/8PSK` 下，每10k个样本分片保存。
