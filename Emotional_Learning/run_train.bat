@echo off
REM 使用 conda test4 环境 (Python 3.12.9) 启动训练
call conda activate test4
python train.py %*
