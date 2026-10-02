#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU LightGBM benchmark node"

apt-get update -y
apt-get install -y python3 python3-pip

# Ubuntu 22.04 ships pip 22.0.2 and has no EXTERNALLY-MANAGED marker, so a plain
# system-wide install works here. Do NOT add --break-system-packages: that flag only
# exists from pip 23.0.1 and this pip rejects it ("no such option"), which aborts the
# install. The GCP equivalent targets Debian 12 (pip 23+) and legitimately needs the flag.
pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle

mkdir -p /home/ubuntu/ml-benchmark
chown ubuntu:ubuntu /home/ubuntu/ml-benchmark

echo "CPU environment ready: lightgbm, scikit-learn, pandas, numpy, kaggle installed system-wide."
