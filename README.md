# PCMA

# 模型训练部分
load_dataset.py 加载数据

model_complex.py 定义模型

generate_sim_dataset.py 建立仿真数据

torch_signal_sim.py 纯torch版本的生成数据，参数可微

train_SignalSeparator.py 训练

train_SignalSeparator_v2.py 添加噪声一致性损失和EVM损失

utiles.py 依赖函数



# 数据分析部分
compensation.py costas环补偿合理性分析

estimate_h.py 估计成型滤波器

split_data.py 裁剪数据

utils_compensation.py 依赖函数


采集数据的原始数据存放在 '/nas/datasets/LYX/PCMA'下，保存格式为I、Q、I、Q…… 每个数据点为一个short

处理后的数据放在 '/nas/datasets/yixin/PCMA/sim_data' 下

