@echo off
REM 折中配置训练（结果保存到 checkpoints_balanced/）
call conda activate test4
python train.py --profile balanced
