import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ScorePredictor:
    def __init__(self, data_path):
        """初始化预测器并加载数据"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载数据并进行初步处理"""
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载成功，共包含 {len(self.data)} 条记录")
        
    def preprocess_data(self):
        """数据预处理和特征工程"""
        # 分离特征和标签
        X = self.data.drop(['姓名', '考试成绩'], axis=1)
        y = self.data['考试成绩']
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"数据预处理完成，训练集大小: {len(self.X_train)}，测试集大小: {len(self.X_test)}")
        
    def explore_data(self):
        """数据探索和可视化"""
        print("\n数据基本信息:")
        print(self.data.info())
        
        print("\n数据统计摘要:")
        print(self.data.describe())
        
        print("\n特征与标签的相关性:")
        corr = self.data.corr()
        print(corr['考试成绩'].sort_values(ascending=False))
        
    def train_multiple_models(self):
        """训练多种模型并选择最佳模型"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # 对于线性模型，使用标准化后的数据
            if name in ['Linear Regression', 'Ridge', 'Lasso']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"\n{name} 模型性能:")
            print(f"均方误差 (MSE): {mse:.4f}")
            print(f"均方根误差 (RMSE): {rmse:.4f}")
            print(f"R² 评分: {r2:.4f}")
        
        # 选择最佳模型
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        self.model = models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        return results
    
    def hyperparameter_tuning(self):
        """对最佳模型进行超参数调优"""
        if self.model is None:
            print("请先训练模型")
            return
        
        if isinstance(self.model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif isinstance(self.model, GradientBoostingRegressor):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif isinstance(self.model, Ridge):
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
        elif isinstance(self.model, Lasso):
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
        else:
            print("该模型不需要超参数调优")
            return
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2')
        
        if isinstance(self.model, (LinearRegression, Ridge, Lasso)):
            grid_search.fit(self.X_train_scaled, self.y_train)
        else:
            grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n超参数调优完成，最佳参数: {grid_search.best_params_}")
        print(f"调优后最佳 R² 评分: {grid_search.best_score_:.4f}")
        
        # 更新模型为调优后的最佳模型
        self.model = grid_search.best_estimator_
    
    def evaluate_model(self):
        """评估最终模型性能"""
        if self.model is None:
            print("请先训练模型")
            return
        
        if isinstance(self.model, (LinearRegression, Ridge, Lasso)):
            y_pred = self.model.predict(self.X_test_scaled)
        else:
            y_pred = self.model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"\n最终模型性能评估:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"R² 评分: {r2:.4f}")
        
        # 可视化预测结果
        self._visualize_results(y_pred)
    
    def _visualize_results(self, y_pred):
        """可视化预测结果"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            # 实际值 vs 预测值散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.xlabel('实际考试成绩')
            plt.ylabel('预测考试成绩')
            plt.title('实际值 vs 预测值')
            plt.savefig('prediction_scatter.png')
            
            # 特征重要性分析（如果模型支持）
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                features = self.X_train.columns
                
                plt.figure(figsize=(12, 8))
                plt.title('特征重要性')
                plt.bar(range(self.X_train.shape[1]), importances[indices])
                plt.xticks(range(self.X_train.shape[1]), [features[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig('feature_importance.png')
                
            print("\n可视化图表已保存为PNG文件")
        except Exception as e:
            print(f"可视化过程中出现错误: {e}")
    
    def predict_new_data(self, new_data):
        """使用训练好的模型预测新数据"""
        if self.model is None:
            print("请先训练模型")
            return None
        
        # 确保新数据格式正确
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        elif not isinstance(new_data, pd.DataFrame):
            print("新数据格式错误")
            return None
        
        # 检查特征列是否匹配
        missing_cols = set(self.X_train.columns) - set(new_data.columns)
        if missing_cols:
            print(f"缺少必要的特征列: {missing_cols}")
            return None
        
        # 选择正确的特征列
        new_data = new_data[self.X_train.columns]
        
        # 标准化（如果模型需要）
        if isinstance(self.model, (LinearRegression, Ridge, Lasso)):
            new_data_scaled = self.scaler.transform(new_data)
            predictions = self.model.predict(new_data_scaled)
        else:
            predictions = self.model.predict(new_data)
        
        return predictions

if __name__ == "__main__":
    # 创建预测器实例
    predictor = ScorePredictor("score_filled.csv")
    
    # 加载和预处理数据
    predictor.load_data()
    predictor.preprocess_data()
    
    # 数据探索（可选）
    # predictor.explore_data()
    
    # 训练并选择最佳模型
    print("\n===== 训练多种模型 =====")
    predictor.train_multiple_models()
    
    # 超参数调优
    print("\n===== 超参数调优 =====")
    predictor.hyperparameter_tuning()
    
    # 评估最终模型
    print("\n===== 评估最终模型 =====")
    predictor.evaluate_model()
    
    print("\n学生考试成绩预测系统已完成")