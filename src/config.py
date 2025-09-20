"""
项目配置文件
包含数据路径、模型参数等配置信息
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 模型路径配置
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# 音频处理参数
AUDIO_PARAMS = {
    'sample_rate': 22050,
    'duration': 30,  # 秒
    'hop_length': 512,
    'n_fft': 2048
}

# 模型训练参数
MODEL_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'validation_split': 0.2
}

# 创建必要的目录
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
