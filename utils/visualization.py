"""
可视化工具模块
提供音乐数据和模型结果的可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MusicVisualizer:
    """音乐数据可视化工具"""
    
    def __init__(self, style='seaborn-v0_8'):
        """初始化可视化器"""
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
        
    def plot_bpm_distribution(self, df, figsize=(12, 8)):
        """绘制BPM分布图"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # BPM直方图
        axes[0, 0].hist(df['tempo'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('🎵 BPM分布直方图', fontweight='bold')
        axes[0, 0].set_xlabel('BPM')
        axes[0, 0].set_ylabel('歌曲数量')
        axes[0, 0].grid(True, alpha=0.3)
        
        # BPM箱线图
        axes[0, 1].boxplot(df['tempo'], vert=True)
        axes[0, 1].set_title('📊 BPM箱线图', fontweight='bold')
        axes[0, 1].set_ylabel('BPM')
        axes[0, 1].grid(True, alpha=0.3)
        
        # BPM密度图
        df['tempo'].plot.density(ax=axes[1, 0], color='coral')
        axes[1, 0].set_title('📈 BPM密度图', fontweight='bold')
        axes[1, 0].set_xlabel('BPM')
        axes[1, 0].grid(True, alpha=0.3)
        
        # BPM分类统计
        bpm_categories = pd.cut(df['tempo'], 
                               bins=[0, 90, 120, 140, 200], 
                               labels=['慢', '中', '快', '极快'])
        category_counts = bpm_categories.value_counts()
        axes[1, 1].pie(category_counts.values, labels=category_counts.index, 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('🎯 BPM类别分布', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print("🎵 BPM统计摘要:")
        print(f"   平均BPM: {df['tempo'].mean():.1f}")
        print(f"   中位数BPM: {df['tempo'].median():.1f}")
        print(f"   标准差: {df['tempo'].std():.1f}")
        print(f"   范围: {df['tempo'].min():.1f} - {df['tempo'].max():.1f}")
    
    def plot_feature_correlation(self, df, figsize=(12, 10)):
        """绘制特征相关性热力图"""
        plt.figure(figsize=figsize)
        
        # 计算相关系数
        correlation = df.corr()
        
        # 创建掩码（只显示下三角）
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        # 绘制热力图
        sns.heatmap(correlation, 
                    mask=mask,
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={"shrink": .8})
        
        plt.title('🔥 音乐特征相关性热力图', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        return correlation
    
    def plot_feature_vs_bpm(self, df, features=None, figsize=(15, 12)):
        """绘制特征与BPM的关系图"""
        if features is None:
            features = [col for col in df.columns if col != 'tempo']
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                axes[i].scatter(df[feature], df['tempo'], alpha=0.6, color=self.colors[i % len(self.colors)])
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('BPM')
                axes[i].set_title(f'📊 {feature} vs BPM')
                axes[i].grid(True, alpha=0.3)
                
                # 添加回归线
                z = np.polyfit(df[feature], df['tempo'], 1)
                p = np.poly1d(z)
                axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
        
        # 隐藏多余的子图
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_scatter_matrix(self, df):
        """创建交互式散点图矩阵"""
        # 选择数值特征
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        fig = px.scatter_matrix(df, 
                               dimensions=numeric_features,
                               color='tempo',
                               title="🎵 音乐特征交互式散点图矩阵",
                               color_continuous_scale='Viridis')
        
        fig.update_layout(height=800)
        fig.show()
        
        return fig
    
    def plot_feature_importance(self, importance_df, top_n=10, figsize=(10, 8)):
        """绘制特征重要性图"""
        if importance_df is None:
            print("❌ 没有特征重要性数据")
            return
        
        # 选择前N个重要特征
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        
        # 水平柱状图
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('重要性')
        plt.title(f'🎯 Top {top_n} 特征重要性', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def create_model_comparison_dashboard(self, results_df):
        """创建模型比较仪表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('测试 MAE 比较', '测试 R² 比较', '训练 vs 测试 MAE', '交叉验证 MAE'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        models = results_df['Model']
        colors = px.colors.qualitative.Set3[:len(models)]
        
        # 测试 MAE
        fig.add_trace(
            go.Bar(x=models, y=results_df['Test MAE'], name='Test MAE', 
                   marker_color=colors),
            row=1, col=1
        )
        
        # 测试 R²
        fig.add_trace(
            go.Bar(x=models, y=results_df['Test R²'], name='Test R²',
                   marker_color=colors),
            row=1, col=2
        )
        
        # 训练 vs 测试 MAE
        fig.add_trace(
            go.Bar(x=models, y=results_df['Train MAE'], name='Train MAE',
                   marker_color='lightblue'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=results_df['Test MAE'], name='Test MAE',
                   marker_color='orange'),
            row=2, col=1
        )
        
        # CV MAE
        fig.add_trace(
            go.Bar(x=models, y=results_df['CV MAE'], name='CV MAE',
                   marker_color=colors),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🎵 模型性能比较仪表板",
            showlegend=False,
            height=800
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="模型", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_xaxes(title_text="模型", row=1, col=2)
        fig.update_yaxes(title_text="R²", row=1, col=2)
        fig.update_xaxes(title_text="模型", row=2, col=1)
        fig.update_yaxes(title_text="MAE", row=2, col=1)
        fig.update_xaxes(title_text="模型", row=2, col=2)
        fig.update_yaxes(title_text="CV MAE", row=2, col=2)
        
        fig.show()
        return fig
