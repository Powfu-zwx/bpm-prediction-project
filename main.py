"""
🎵 音乐BPM预测项目主程序
使用机器学习技术预测音乐的每分钟节拍数(BPM)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目模块到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from utils.visualization import MusicVisualizer
from src.config import *

def main():
    """主函数"""
    print("🎵" + "="*60)
    print("🎶  欢迎使用音乐BPM预测系统")
    print("🎵" + "="*60)
    
    # 初始化组件
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    visualizer = MusicVisualizer()
    
    print("\n📊 第一步：数据生成和预处理")
    print("-" * 40)
    
    # 生成示例数据
    df = data_processor.generate_sample_data(n_samples=1000)
    print(f"✅ 生成 {len(df)} 首歌曲的音乐特征数据")
    
    # 数据清洗
    df_clean = data_processor.clean_data(df)
    
    # 特征工程
    df_engineered = data_processor.feature_engineering(df_clean)
    
    # 保存处理后的数据
    processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'music_features.csv')
    df_engineered.to_csv(processed_data_path, index=False)
    print(f"✅ 处理后的数据已保存到: {processed_data_path}")
    
    print("\n🎯 第二步：数据可视化分析")
    print("-" * 40)
    
    # BPM分布分析
    visualizer.plot_bpm_distribution(df_engineered)
    
    # 特征相关性分析
    correlation_matrix = visualizer.plot_feature_correlation(df_engineered)
    
    print("\n🔧 第三步：数据预处理")
    print("-" * 40)
    
    # 准备训练和测试数据
    X_train, X_test, y_train, y_test = data_processor.prepare_data(df_engineered)
    
    # 特征选择
    X_train_selected, selected_features = data_processor.select_features(X_train, y_train, k=8)
    X_test_selected = data_processor.feature_selector.transform(X_test)
    
    # 特征标准化
    X_train_scaled, X_test_scaled = data_processor.scale_features(X_train_selected, X_test_selected)
    
    # 保存预处理器
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    data_processor.save_preprocessor(preprocessor_path)
    
    print("\n🤖 第四步：模型训练")
    print("-" * 40)
    
    # 初始化并训练所有模型
    model_trainer.initialize_models()
    results_df = model_trainer.train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 显示模型比较结果
    print("\n📊 模型性能比较:")
    print(results_df.round(3))
    
    # 绘制模型比较图
    model_trainer.plot_model_comparison()
    
    print("\n⚡ 第五步：超参数调优")
    print("-" * 40)
    
    # 对最佳模型进行超参数调优
    best_params = model_trainer.hyperparameter_tuning(X_train_scaled, y_train)
    if best_params:
        print(f"✅ 超参数调优完成")
    
    # 保存最佳模型
    best_model_path = os.path.join(MODELS_DIR, 'best_bpm_model.joblib')
    model_trainer.save_best_model(best_model_path)
    
    print("\n📈 第六步：模型评估")
    print("-" * 40)
    
    # 评估最佳模型
    evaluation_results = evaluator.evaluate_model(
        model_trainer.best_model, 
        X_test_scaled, 
        y_test, 
        model_trainer.best_model_name
    )
    
    # 生成详细评估报告
    evaluator.generate_evaluation_report(model_trainer.best_model_name)
    
    # 可视化评估结果
    evaluator.plot_predictions_vs_actual(model_trainer.best_model_name)
    evaluator.plot_residuals(model_trainer.best_model_name)
    evaluator.plot_error_distribution(model_trainer.best_model_name)
    
    print("\n🎯 第七步：特征重要性分析")
    print("-" * 40)
    
    # 特征重要性分析
    feature_importance = model_trainer.get_feature_importance(selected_features)
    if feature_importance is not None:
        print("\n🏆 Top 5 重要特征:")
        print(feature_importance.head())
        
        # 可视化特征重要性
        visualizer.plot_feature_importance(feature_importance)
    
    print("\n🎉 第八步：项目总结")
    print("-" * 40)
    
    print("✅ 项目完成情况:")
    print(f"   📊 数据集大小: {len(df_engineered)} 首歌曲")
    print(f"   🔧 特征数量: {len(selected_features)} 个")
    print(f"   🤖 最佳模型: {model_trainer.best_model_name}")
    print(f"   🎯 测试MAE: {evaluation_results['MAE']:.2f} BPM")
    print(f"   📈 测试R²: {evaluation_results['R²']:.3f}")
    
    print(f"\n💾 模型文件保存位置:")
    print(f"   🤖 最佳模型: {best_model_path}")
    print(f"   🔧 预处理器: {preprocessor_path}")
    print(f"   📊 处理后数据: {processed_data_path}")
    
    print("\n🎵 感谢使用BPM预测系统！")
    print("🎵" + "="*60)

def predict_bpm(audio_features, model_path=None, preprocessor_path=None):
    """
    使用训练好的模型预测单首歌曲的BPM
    
    Args:
        audio_features (dict): 音频特征字典
        model_path (str): 模型文件路径
        preprocessor_path (str): 预处理器文件路径
    
    Returns:
        float: 预测的BPM值
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'best_bmp_model.joblib')
    if preprocessor_path is None:
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    
    # 加载模型和预处理器
    model_trainer = ModelTrainer()
    data_processor = DataProcessor()
    
    model_trainer.load_model(model_path)
    data_processor.load_preprocessor(preprocessor_path)
    
    # 预处理特征
    features_df = pd.DataFrame([audio_features])
    features_selected = data_processor.feature_selector.transform(features_df)
    features_scaled = data_processor.scaler.transform(features_selected)
    
    # 预测
    predicted_bpm = model_trainer.best_model.predict(features_scaled)[0]
    
    return predicted_bpm

if __name__ == "__main__":
    # 设置随机种子以保证结果可重现
    np.random.seed(42)
    
    # 运行主程序
    main()
    
    # 示例：预测单首歌曲的BPM
    print("\n🎵 示例：预测新歌曲BPM")
    print("-" * 30)
    
    sample_features = {
        'danceability': 0.8,
        'energy': 0.9,
        'loudness': -5.0,
        'speechiness': 0.1,
        'acousticness': 0.2,
        'instrumentalness': 0.0,
        'liveness': 0.3,
        'valence': 0.7,
        'duration_ms': 180000
    }
    
    try:
        predicted_bpm = predict_bpm(sample_features)
        print(f"🎯 预测BPM: {predicted_bpm:.1f}")
        
        # 根据BPM给出音乐类型建议
        if predicted_bpm < 90:
            music_type = "慢歌/抒情"
        elif predicted_bpm < 120:
            music_type = "流行/摇滚"
        elif predicted_bpm < 140:
            music_type = "舞曲/电子"
        else:
            music_type = "快节奏/极速"
        
        print(f"🎼 音乐类型: {music_type}")
        
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")
        print("请先运行主程序训练模型")
