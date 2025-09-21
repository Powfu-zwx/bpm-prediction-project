"""
模型评估模块
提供全面的模型性能评估和可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """综合评估模型性能"""
        print(f"📊 评估模型: {model_name}")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # 计算残差
        residuals = y_test - y_pred
        
        # 保存结果
        self.evaluation_results[model_name] = {
            'predictions': y_pred,
            'actual': y_test,
            'residuals': residuals,
            'metrics': {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
        }
        
        # 打印结果
        print(f"   📈 MAE: {mae:.2f} BPM")
        print(f"   📈 RMSE: {rmse:.2f} BPM")
        print(f"   📈 R²: {r2:.3f}")
        print(f"   📈 MAPE: {mape:.1%}")
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'predictions': y_pred
        }
    
    def plot_predictions_vs_actual(self, model_name="Model"):
        """绘制预测值vs实际值散点图"""
        if model_name not in self.evaluation_results:
            print(f"❌ 没有找到模型 {model_name} 的评估结果")
            return
        
        results = self.evaluation_results[model_name]
        y_test = results['actual']
        y_pred = results['predictions']
        
        plt.figure(figsize=(10, 8))
        
        # 散点图
        plt.scatter(y_test, y_pred, alpha=0.6, color='skyblue', edgecolors='navy')
        
        # 完美预测线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
        
        # 格式化
        plt.xlabel('实际BPM', fontsize=12)
        plt.ylabel('预测BPM', fontsize=12)
        plt.title(f'🎯 {model_name} - 预测值 vs 实际值', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加R²信息
        r2 = results['metrics']['R²']
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, model_name="Model"):
        """绘制残差图"""
        if model_name not in self.evaluation_results:
            print(f"❌ 没有找到模型 {model_name} 的评估结果")
            return
        
        results = self.evaluation_results[model_name]
        y_pred = results['predictions']
        residuals = results['residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 残差vs预测值
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('预测值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('📊 残差 vs 预测值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='lightcoral')
        axes[0, 1].set_xlabel('残差')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('📈 残差分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q图（检验正态性）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('📊 残差Q-Q图')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差绝对值vs预测值
        axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.6, color='orange')
        axes[1, 1].set_xlabel('预测值')
        axes[1, 1].set_ylabel('|残差|')
        axes[1, 1].set_title('📊 绝对残差 vs 预测值')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_distribution(self, model_name="Model"):
        """绘制误差分布图"""
        if model_name not in self.evaluation_results:
            print(f"❌ 没有找到模型 {model_name} 的评估结果")
            return
        
        results = self.evaluation_results[model_name]
        y_test = results['actual']
        y_pred = results['predictions']
        errors = np.abs(y_test - y_pred)
        
        plt.figure(figsize=(12, 8))
        
        # 创建误差区间
        bins = [0, 2, 5, 10, 15, 20, float('inf')]
        labels = ['≤2', '2-5', '5-10', '10-15', '15-20', '>20']
        
        error_counts = []
        for i in range(len(bins)-1):
            if i == len(bins)-2:  # 最后一个区间
                count = ((errors > bins[i]) & (errors <= bins[i+1])).sum()
            else:
                count = ((errors > bins[i]) & (errors <= bins[i+1])).sum()
            error_counts.append(count)
        
        # 绘制柱状图
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))
        bars = plt.bar(labels, error_counts, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加百分比标签
        total = sum(error_counts)
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/total:.1%})',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('误差范围 (BPM)', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.title(f'🎯 {model_name} - 预测误差分布', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加统计信息
        mae = results['metrics']['MAE']
        plt.text(0.7, 0.9, f'平均绝对误差: {mae:.2f} BPM', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, model_name="Model"):
        """创建交互式仪表板"""
        if model_name not in self.evaluation_results:
            print(f"❌ 没有找到模型 {model_name} 的评估结果")
            return
        
        results = self.evaluation_results[model_name]
        y_test = results['actual']
        y_pred = results['predictions']
        residuals = results['residuals']
        metrics = results['metrics']
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('预测值 vs 实际值', '残差分布', '误差vs预测值', '性能指标'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 预测值vs实际值
        fig.add_trace(
            go.Scatter(x=y_test, y=y_pred, mode='markers', name='预测点',
                      marker=dict(size=6, opacity=0.6, color='blue')),
            row=1, col=1
        )
        
        # 完美预测线
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='完美预测', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 残差直方图
        fig.add_trace(
            go.Histogram(x=residuals, name='残差分布', nbinsx=30),
            row=1, col=2
        )
        
        # 误差vs预测值
        fig.add_trace(
            go.Scatter(x=y_pred, y=np.abs(residuals), mode='markers', name='绝对误差',
                      marker=dict(size=6, opacity=0.6, color='orange')),
            row=2, col=1
        )
        
        # 性能指标
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())
        fig.add_trace(
            go.Bar(x=metrics_names, y=metrics_values, name='性能指标',
                   marker=dict(color=['blue', 'green', 'orange', 'red', 'purple'])),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text=f"🎵 {model_name} - 模型性能仪表板",
            showlegend=False,
            height=800
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="实际BPM", row=1, col=1)
        fig.update_yaxes(title_text="预测BPM", row=1, col=1)
        fig.update_xaxes(title_text="残差", row=1, col=2)
        fig.update_yaxes(title_text="频次", row=1, col=2)
        fig.update_xaxes(title_text="预测BPM", row=2, col=1)
        fig.update_yaxes(title_text="绝对误差", row=2, col=1)
        fig.update_xaxes(title_text="指标", row=2, col=2)
        fig.update_yaxes(title_text="值", row=2, col=2)
        
        fig.show()
        
        return fig
    
    def generate_evaluation_report(self, model_name="Model"):
        """生成评估报告"""
        if model_name not in self.evaluation_results:
            print(f"❌ 没有找到模型 {model_name} 的评估结果")
            return
        
        results = self.evaluation_results[model_name]
        metrics = results['metrics']
        
        print("="*60)
        print(f"🎵 {model_name} - 模型性能评估报告")
        print("="*60)
        
        print("\n📊 核心性能指标:")
        print(f"   🎯 平均绝对误差 (MAE): {metrics['MAE']:.2f} BPM")
        print(f"   📐 均方根误差 (RMSE): {metrics['RMSE']:.2f} BPM")
        print(f"   📈 决定系数 (R²): {metrics['R²']:.3f}")
        print(f"   📊 平均绝对百分比误差 (MAPE): {metrics['MAPE']:.1%}")
        
        # 性能解释
        print("\n📝 性能解释:")
        if metrics['R²'] >= 0.9:
            print("   ✅ 优秀: 模型解释了90%以上的方差")
        elif metrics['R²'] >= 0.8:
            print("   🟡 良好: 模型解释了80-90%的方差")
        elif metrics['R²'] >= 0.7:
            print("   🟠 一般: 模型解释了70-80%的方差")
        else:
            print("   🔴 需要改进: 模型解释的方差不足70%")
        
        if metrics['MAE'] <= 5:
            print("   ✅ MAE ≤ 5 BPM: 预测精度很高")
        elif metrics['MAE'] <= 10:
            print("   🟡 MAE ≤ 10 BPM: 预测精度良好")
        else:
            print("   🔴 MAE > 10 BPM: 预测精度需要改进")
        
        print("\n🎯 应用建议:")
        if metrics['MAE'] <= 5 and metrics['R²'] >= 0.8:
            print("   ✅ 该模型可以用于生产环境")
        elif metrics['MAE'] <= 10 and metrics['R²'] >= 0.7:
            print("   🟡 该模型可以用于原型开发，但需要进一步优化")
        else:
            print("   🔴 建议重新训练或尝试其他算法")
        
        print("="*60)
