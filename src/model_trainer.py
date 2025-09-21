"""
模型训练模块
实现多种机器学习模型用于BPM预测
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from .config import *

class ModelTrainer:
    """BPM预测模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_results = {}
        
    def initialize_models(self):
        """初始化多个机器学习模型"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=MODEL_PARAMS['random_state'],
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=MODEL_PARAMS['random_state']
            ),
            'Support Vector Regression': SVR(kernel='rbf'),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Multi-layer Perceptron': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                random_state=MODEL_PARAMS['random_state'],
                max_iter=1000
            )
        }
        print(f"✅ 初始化了 {len(self.models)} 个模型")
        
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """训练所有模型并评估性能"""
        print("🚀 开始训练所有模型...")
        
        results = []
        
        for name, model in self.models.items():
            print(f"📊 训练 {name}...")
            
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # 计算评估指标
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # 交叉验证
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                
                result = {
                    'Model': name,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae,
                    'Train RMSE': train_rmse,
                    'Test RMSE': test_rmse,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'CV MAE': cv_mae
                }
                
                results.append(result)
                self.model_results[name] = {
                    'model': model,
                    'metrics': result,
                    'predictions': {
                        'train': y_pred_train,
                        'test': y_pred_test
                    }
                }
                
                print(f"   ✅ Test MAE: {test_mae:.2f}, Test R²: {test_r2:.3f}")
                
            except Exception as e:
                print(f"   ❌ {name} 训练失败: {str(e)}")
                
        # 创建结果DataFrame并排序
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('Test MAE')
        
        # 选择最佳模型
        best_result = self.results_df.iloc[0]
        self.best_model_name = best_result['Model']
        self.best_model = self.model_results[self.best_model_name]['model']
        
        print(f"\n🏆 最佳模型: {self.best_model_name}")
        print(f"   Test MAE: {best_result['Test MAE']:.2f}")
        print(f"   Test R²: {best_result['Test R²']:.3f}")
        
        return self.results_df
    
    def hyperparameter_tuning(self, X_train, y_train, model_name=None):
        """超参数调优"""
        if model_name is None:
            model_name = self.best_model_name
            
        print(f"🔧 对 {model_name} 进行超参数调优...")
        
        # 定义超参数网格
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Support Vector Regression': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1],
                'epsilon': [0.01, 0.1, 1]
            }
        }
        
        if model_name in param_grids:
            model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            # 网格搜索
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=5, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 更新最佳模型
            self.best_model = grid_search.best_estimator_
            
            print(f"✅ 最佳参数: {grid_search.best_params_}")
            print(f"✅ 最佳CV得分: {-grid_search.best_score_:.2f}")
            
            return grid_search.best_params_
        else:
            print(f"❌ {model_name} 没有定义超参数网格")
            return None
    
    def get_feature_importance(self, feature_names):
        """获取特征重要性"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            print("❌ 当前模型不支持特征重要性分析")
            return None
    
    def plot_model_comparison(self):
        """绘制模型性能比较图"""
        if self.results_df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Test MAE比较
            axes[0, 0].barh(self.results_df['Model'], self.results_df['Test MAE'])
            axes[0, 0].set_title('🎯 Test MAE (越小越好)')
            axes[0, 0].set_xlabel('MAE')
            
            # Test R²比较
            axes[0, 1].barh(self.results_df['Model'], self.results_df['Test R²'])
            axes[0, 1].set_title('📊 Test R² (越大越好)')
            axes[0, 1].set_xlabel('R²')
            
            # Train vs Test MAE
            x = np.arange(len(self.results_df))
            width = 0.35
            axes[1, 0].bar(x - width/2, self.results_df['Train MAE'], width, label='Train MAE')
            axes[1, 0].bar(x + width/2, self.results_df['Test MAE'], width, label='Test MAE')
            axes[1, 0].set_title('🔄 Train vs Test MAE')
            axes[1, 0].set_xlabel('Models')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(self.results_df['Model'], rotation=45)
            axes[1, 0].legend()
            
            # CV MAE
            axes[1, 1].barh(self.results_df['Model'], self.results_df['CV MAE'])
            axes[1, 1].set_title('🔀 Cross-Validation MAE')
            axes[1, 1].set_xlabel('CV MAE')
            
            plt.tight_layout()
            plt.show()
    
    def save_best_model(self, filepath):
        """保存最佳模型"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'results': self.results_df
            }, filepath)
            print(f"✅ 最佳模型已保存到: {filepath}")
        else:
            print("❌ 没有训练好的模型可保存")
    
    def load_model(self, filepath):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        if 'results' in model_data:
            self.results_df = model_data['results']
        print(f"✅ 模型 {self.best_model_name} 已从 {filepath} 加载")
