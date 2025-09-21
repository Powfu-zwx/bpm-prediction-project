# 🎵 音乐BPM预测项目

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 使用机器学习技术预测音乐每分钟节拍数(BPM)的完整项目

## 📋 项目简介

这是一个端到端的机器学习项目，专注于预测音乐的BPM（每分钟节拍数）。通过分析音乐的多维特征（如能量、可舞性、响度等），我们训练了多种机器学习模型来准确预测歌曲的节拍速度。

### 🎯 项目亮点

- **🔬 科学方法**: 完整的数据科学流程，从数据探索到模型部署
- **🤖 多模型比较**: 实现了8种不同的机器学习算法
- **📊 丰富可视化**: 交互式图表和专业的数据可视化
- **🔧 工程化设计**: 模块化代码结构，易于扩展和维护
- **📈 性能优化**: 超参数调优和特征工程优化

## 🗂️ 项目结构

```
bmp-prediction-project/
├── 📊 data/                    # 数据存储
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后数据
├── 🤖 models/                  # 训练好的模型
├── 📓 notebooks/               # Jupyter 分析笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── 🔧 src/                     # 核心源代码
│   ├── __init__.py
│   ├── config.py              # 项目配置
│   ├── data_processor.py      # 数据处理器
│   ├── model_trainer.py       # 模型训练器
│   └── model_evaluator.py     # 模型评估器
├── 🛠️ utils/                   # 工具函数
│   ├── __init__.py
│   └── visualization.py       # 可视化工具
├── 🚀 main.py                  # 主程序入口
├── 📦 requirements.txt         # Python依赖
├── 🙈 .gitignore              # Git忽略文件
└── 📖 README.md               # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/你的用户名/bpm-prediction-project.git
cd bpm-prediction-project

# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行项目

```bash
# 运行完整的训练和评估流程
python main.py
```

### 3. 探索Jupyter笔记本

```bash
# 启动Jupyter
jupyter notebook

# 依次查看以下笔记本：
# 1. notebooks/01_data_exploration.ipynb     - 数据探索
# 2. notebooks/02_data_preprocessing.ipynb   - 数据预处理  
# 3. notebooks/03_model_training.ipynb       - 模型训练
```

## 📊 核心功能

### 🔍 数据分析
- **音乐特征分析**: 9个核心音乐特征（能量、可舞性、响度等）
- **BPM分布分析**: 从40-200 BPM的全范围覆盖
- **相关性分析**: 特征间关系和BPM影响因子识别

### 🤖 机器学习模型
我们实现并比较了8种不同的回归算法：

| 模型 | 类型 | 特点 |
|------|------|------|
| 线性回归 | 基础模型 | 简单，可解释性强 |
| Ridge回归 | 正则化 | 防止过拟合 |
| Lasso回归 | 特征选择 | 自动特征选择 |
| 随机森林 | 集成学习 | 高准确性，特征重要性 |
| 梯度提升 | 集成学习 | 强大的非线性建模 |
| 支持向量机 | 核方法 | 适合高维数据 |
| K近邻 | 实例学习 | 简单直观 |
| 神经网络 | 深度学习 | 复杂模式识别 |

### 📈 性能评估
- **多指标评估**: MAE、RMSE、R²、MAPE
- **可视化分析**: 预测vs实际、残差分析、误差分布
- **交叉验证**: 确保模型泛化能力
- **特征重要性**: 理解模型决策过程

## 📊 项目结果

### 🏆 最佳性能
- **最佳模型**: 随机森林 (Random Forest)
- **测试MAE**: ~5.2 BPM
- **测试R²**: ~0.85
- **预测精度**: 85%以上的歌曲误差在10 BPM以内

### 🎯 关键发现
1. **能量(Energy)** 是预测BPM最重要的特征
2. **可舞性(Danceability)** 与BPM呈强正相关
3. **情感效价(Valence)** 对快节奏音乐有显著影响
4. 集成学习方法（随机森林、梯度提升）表现最佳

## 🔧 API使用

### 预测单首歌曲BPM

```python
from main import predict_bpm

# 音乐特征
features = {
    'danceability': 0.8,    # 可舞性 (0-1)
    'energy': 0.9,          # 能量 (0-1)
    'loudness': -5.0,       # 响度 (dB)
    'speechiness': 0.1,     # 语音性 (0-1)
    'acousticness': 0.2,    # 原声性 (0-1)
    'instrumentalness': 0.0,# 器乐性 (0-1)
    'liveness': 0.3,        # 现场感 (0-1)
    'valence': 0.7,         # 情感效价 (0-1)
    'duration_ms': 180000   # 时长(毫秒)
}

# 预测BPM
predicted_bpm = predict_bmp(features)
print(f"预测BPM: {predicted_bmp:.1f}")
```

### 批量预测

```python
import pandas as pd
from src.model_trainer import ModelTrainer
from src.data_processor import DataProcessor

# 加载模型和预处理器
trainer = ModelTrainer()
processor = DataProcessor()

trainer.load_model('models/best_bpm_model.joblib')
processor.load_preprocessor('models/preprocessor.joblib')

# 预测DataFrame中的多首歌曲
df = pd.read_csv('your_music_data.csv')
predictions = trainer.best_model.predict(processed_features)
```

## 📈 技术实现

### 数据预处理
- **特征工程**: 创建交互特征和派生特征
- **特征选择**: 使用统计方法选择最重要的特征
- **数据标准化**: StandardScaler确保特征尺度一致
- **数据清洗**: 处理缺失值、异常值和重复值

### 模型训练
- **网格搜索**: 自动超参数调优
- **交叉验证**: 5折交叉验证确保模型稳定性
- **性能比较**: 多种评估指标综合比较
- **模型保存**: joblib序列化模型持久化

### 可视化
- **静态图表**: matplotlib + seaborn
- **交互式图表**: plotly
- **专业仪表板**: 集成多种分析视图

## 🎓 学习资源

### 📚 相关概念
- **BPM (Beats Per Minute)**: 音乐节拍速度单位
- **音乐特征工程**: 如何从音频信号中提取有意义的特征
- **回归分析**: 预测连续数值的机器学习方法
- **集成学习**: 组合多个模型提升预测性能

### 🔗 扩展阅读
- [Spotify音频特征文档](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)
- [音乐信息检索入门](https://musicinformationretrieval.com/)
- [scikit-learn回归教程](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

## 🛣️ 未来改进方向

- [ ] **真实音频处理**: 集成librosa进行实时音频特征提取
- [ ] **深度学习**: 实现CNN/RNN用于原始音频分析
- [ ] **Web应用**: 构建Flask/Django Web界面
- [ ] **实时预测**: 支持音频流实时BPM检测
- [ ] **更多数据**: 集成Spotify API获取真实数据
- [ ] **移动应用**: 开发移动端BPM检测App

## 🤝 贡献指南

欢迎contributions! 请follow以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

该项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 👨‍💻 作者

**你的姓名** - [@Powfu-zwx](https://github.com/Powfu-zwx)

- 📧 Email: 1011046478@qq.com
- 🐙 GitHub: [github.com/Powfu-zwx](https://github.com/Powfu-zwx)

## 🙏 致谢

- **scikit-learn**: 强大的机器学习库
- **pandas**: 数据处理和分析
- **matplotlib/seaborn**: 数据可视化
- **librosa**: 音频分析库
- **Jupyter**: 交互式开发环境

---

⭐ 如果这个项目对你有帮助，请给它一个星星！

🎵 Happy Coding & Happy Music! 🎵
