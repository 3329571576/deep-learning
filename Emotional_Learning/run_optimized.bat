@echo off
REM 使用优化配置重新训练（结果保存到 checkpoints_optimized/）
call conda activate test4
python train.py --profile optimized
