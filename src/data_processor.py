"""
数据处理模块
负责数据加载、清洗、预处理和特征工程
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os
from .config import *

class DataProcessor:
    """音乐数据处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        
    def generate_sample_data(self, n_samples=1000, random_state=42):
        """生成示例音乐数据"""
        np.random.seed(random_state)
        
        # 创建音乐特征
        data = {
            'danceability': np.random.beta(2, 5, n_samples),
            'energy': np.random.beta(2, 2, n_samples),
            'loudness': np.random.normal(-10, 5, n_samples),
            'speechiness': np.random.beta(1, 9, n_samples),
            'acousticness': np.random.beta(1, 3, n_samples),
            'instrumentalness': np.random.beta(1, 9, n_samples),
            'liveness': np.random.beta(1, 9, n_samples),
            'valence': np.random.beta(2, 2, n_samples),
            'duration_ms': np.random.normal(200000, 50000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # 确保数值在合理范围内
        df['loudness'] = np.clip(df['loudness'], -60, 0)
        df['duration_ms'] = np.clip(df['duration_ms'], 30000, 600000)
        
        # 生成BPM（基于特征的复杂关系）
        df['tempo'] = (
            60 +
            df['energy'] * 80 +
            df['danceability'] * 60 +
            df['valence'] * 30 +
            np.random.normal(0, 15, n_samples)
        )
        
        df['tempo'] = np.clip(df['tempo'], 40, 200)
        
        return df
    
    def clean_data(self, df):
        """数据清洗"""
        print("🧹 开始数据清洗...")
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"发现 {missing_values.sum()} 个缺失值")
            df = df.dropna()
        
        # 检查重复值
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"删除 {duplicates} 个重复行")
            df = df.drop_duplicates()
            
        # 检查异常值（使用IQR方法）
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"{col}: 发现 {outliers} 个异常值")
                
        print(f"✅ 数据清洗完成！最终数据形状: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """特征工程"""
        print("🔧 开始特征工程...")
        
        # 创建新特征
        df['duration_min'] = df['duration_ms'] / 60000  # 转换为分钟
        df['energy_danceability'] = df['energy'] * df['danceability']  # 交互特征
        df['loudness_energy'] = df['loudness'] * df['energy']  # 响度-能量交互
        
        # 创建分类特征
        df['tempo_category'] = pd.cut(df['tempo'], 
                                    bins=[0, 90, 120, 140, 200], 
                                    labels=['慢', '中', '快', '极快'])
        
        # 对分类特征进行独热编码
        tempo_dummies = pd.get_dummies(df['tempo_category'], prefix='tempo_cat')
        df = pd.concat([df, tempo_dummies], axis=1)
        df = df.drop('tempo_category', axis=1)
        
        print(f"✅ 特征工程完成！新特征数量: {df.shape[1]}")
        return df
    
    def select_features(self, X, y, k=10):
        """特征选择"""
        print(f"🎯 选择前 {k} 个最重要的特征...")
        
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # 获取选中的特征名称
        feature_names = X.columns
        selected_mask = self.feature_selector.get_support()
        self.selected_features = feature_names[selected_mask].tolist()
        
        print(f"✅ 选中的特征: {self.selected_features}")
        return X_selected, self.selected_features
    
    def scale_features(self, X_train, X_test=None):
        """特征标准化"""
        print("📊 进行特征标准化...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        print("🎲 分割训练和测试数据...")
        
        # 分离特征和目标变量
        X = df.drop('tempo', axis=1)
        y = df['tempo']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """保存预处理器"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }, filepath)
        print(f"✅ 预处理器已保存到: {filepath}")
    
    def load_preprocessor(self, filepath):
        """加载预处理器"""
        preprocessor = joblib.load(filepath)
        self.scaler = preprocessor['scaler']
        self.feature_selector = preprocessor['feature_selector']
        self.selected_features = preprocessor['selected_features']
        print(f"✅ 预处理器已从 {filepath} 加载")
