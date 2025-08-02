#!/bin/bash
source /data/projects/crypto-ai/venv/bin/activate

python3.8 /data/projects/crypto-ai/coin_minute_his.py  >> /data/projects/crypto-ai/log/kline_fetch.log 2>&1

free -m