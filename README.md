# PCMA

运行前在当前工作区创建临时文件夹
```bash
mkdir -p ./src/check_points
mkdir -p ./src/pics
mkdir -p ./src/splited_data
```

# 模型训练部分
load_dataset.py 加载数据
model_complex.py 定义模型
generate_sim_dataset_test.py 创建测试集
generate_sim_dataset.py 创建训练/验证集
train_SignalSeparator.py 训练
utiles.py 依赖函数

# 数据分析部分
compensation.py costas环补偿合理性分析
estimate_h.py 估计成型滤波器
split_data.py 裁剪数据
utils_compensation.py 依赖函数


采集数据的原始数据存放在 '/nas/datasets/LYX/PCMA'下，保存格式为I、Q、I、Q…… 每个数据点为一个short

处理后的数据放在 '/nas/datasets/yixin/PCMA/sim_data' 下

启动命令： 
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun --nproc_per_node=7 train_SignalSeparator_improved.py \
  --batch_size 64 --epochs 80 --N 100000 \
  --lr 2e-4 --weight_decay 1e-4 \
  --warmup_steps 2000 --min_lr_ratio 0.1 \
  --accum_steps 1 --ema_decay 0.999 --use_ema \
  > logN100k.txt


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun --nproc_per_node=7 train_SignalSeparator_improved.py \
  --batch_size 64 --epochs 80


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --nproc_per_node=6 test_all_ddp.py \
  --ckpt_path "./src/check_points/signal_separator_qpsk_train_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.30,0.90]_snrU[12,30]_N100000_varsnr_ampr_phi1phi2_c64_best.pth" \
  --test_data_path "/nas/datasets/yixin/PCMA/sim_data/qpsk_test_all_grid_F110_F210_P18_P28_A5_S4_R1.pth" \
  --out_dir "./src/pics/test_all_viz_ddp" \
  --batch_size 64 --num_workers 2 --amp

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
torchrun --nproc_per_node=1 test_all_ddp.py \
  --ckpt_path "./src/check_points/signal_separator_qpsk_train_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.30,0.90]_snrU[12,30]_N100000_varsnr_ampr_phi1phi2_c64_epoch80.pth" \
  --test_data_path "/nas/datasets/yixin/PCMA/sim_data/qpsk_test_all_grid_F110_F210_P18_P28_A5_S4_R1.pth" \
  --out_dir "./src/pics/test_all_newdemod" \
  --batch_size 64 --num_workers 4 --amp

CUDA_VISIBLE_DEVICES=0,2,4 \
python test_all_ddp_newdemod.py \
  --ckpt_path "./src/check_points/all/signal_separator_qpsk_train_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.30,0.90]_snrU[12,30]_N100000_varsnr_ampr_phi1phi2_delay0T_c64_best.pth" \
  --test_data_path "/nas/datasets/yixin/PCMA/sim_data/qpsk_test_all_grid_F110_F210_P18_P28_A5_S4_R1.pth" \
  --out_dir "./src/newdemod2" \
  --batch_size 64 --num_workers 4 --amp

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
torchrun --nproc_per_node=7 train_SignalSeparator_improved.py \
  --batch_size 64 --epochs 120 --N 100000 \
  --lr 5e-4 \
  --warmup_epochs 3 --min_lr_ratio 0.1 \
  --accum_steps 1 --ema_decay 0.999 --use_ema \
  --mse_epoch 120 
```
