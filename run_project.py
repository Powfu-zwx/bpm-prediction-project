#!/usr/bin/env python3
"""
🎵 BPM预测项目快速运行脚本
这个脚本提供了多种运行模式，方便用户快速体验项目功能
"""

import argparse
import sys
import os

def run_full_pipeline():
    """运行完整的训练评估流程"""
    print("🚀 运行完整的BPM预测流程...")
    from main import main
    main()

def run_demo_prediction():
    """运行演示预测"""
    print("🎵 运行BPM预测演示...")
    from main import predict_bpm
    
    # 示例歌曲特征
    examples = [
        {
            'name': '快节奏舞曲',
            'features': {
                'danceability': 0.9,
                'energy': 0.95,
                'loudness': -3.0,
                'speechiness': 0.05,
                'acousticness': 0.1,
                'instrumentalness': 0.8,
                'liveness': 0.2,
                'valence': 0.8,
                'duration_ms': 200000
            }
        },
        {
            'name': '慢抒情歌',
            'features': {
                'danceability': 0.3,
                'energy': 0.2,
                'loudness': -12.0,
                'speechiness': 0.1,
                'acousticness': 0.8,
                'instrumentalness': 0.1,
                'liveness': 0.1,
                'valence': 0.3,
                'duration_ms': 240000
            }
        },
        {
            'name': '流行摇滚',
            'features': {
                'danceability': 0.6,
                'energy': 0.7,
                'loudness': -6.0,
                'speechiness': 0.1,
                'acousticness': 0.3,
                'instrumentalness': 0.0,
                'liveness': 0.3,
                'valence': 0.6,
                'duration_ms': 210000
            }
        }
    ]
    
    try:
        for example in examples:
            predicted_bpm = predict_bpm(example['features'])
            print(f"🎼 {example['name']}: {predicted_bpm:.1f} BPM")
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")
        print("请先运行完整流程训练模型: python run_project.py --mode full")

def run_jupyter():
    """启动Jupyter笔记本"""
    print("📓 启动Jupyter笔记本...")
    os.system("jupyter notebook notebooks/")

def show_project_info():
    """显示项目信息"""
    print("🎵" + "="*60)
    print("🎶  音乐BPM预测项目")
    print("🎵" + "="*60)
    print()
    print("📊 项目功能:")
    print("   🔍 音乐特征分析和数据探索")
    print("   🤖 8种机器学习模型训练和比较")
    print("   📈 模型性能评估和可视化")
    print("   🎯 BPM预测和音乐分类")
    print()
    print("🚀 运行模式:")
    print("   --mode full      运行完整训练评估流程")
    print("   --mode demo      运行预测演示")
    print("   --mode jupyter   启动Jupyter笔记本")
    print("   --mode info      显示项目信息")
    print()
    print("📁 项目结构:")
    print("   📊 data/         数据文件")
    print("   🤖 models/       训练好的模型")
    print("   📓 notebooks/    Jupyter分析笔记本")
    print("   🔧 src/          核心源代码")
    print("   🛠️ utils/        工具函数")
    print()
    print("📚 快速开始:")
    print("   1. pip install -r requirements.txt")
    print("   2. python run_project.py --mode full")
    print("   3. python run_project.py --mode demo")
    print()
    print("🔗 更多信息请查看 README.md")
    print("🎵" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description="🎵 BPM预测项目运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_project.py --mode full      # 运行完整流程
  python run_project.py --mode demo      # 运行预测演示
  python run_project.py --mode jupyter   # 启动Jupyter
  python run_project.py --mode info      # 显示项目信息
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'demo', 'jupyter', 'info'],
        default='info',
        help='运行模式 (默认: info)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'demo':
        run_demo_prediction()
    elif args.mode == 'jupyter':
        run_jupyter()
    elif args.mode == 'info':
        show_project_info()

if __name__ == "__main__":
    main()
