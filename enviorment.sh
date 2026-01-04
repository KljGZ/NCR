echo "Install pytorch (CUDA 11.3, 官方源)"
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
