import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

class RegressionMethodAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        self.kf = KFold(n_splits=10) # 10-fold cross-validation
    
    # Load data from file
    def load_data(self):
        self.data = pd.read_csv(self.data_path, delimiter="\t")
        self.X = self.data.drop(columns=["InstanceID"])  # Features
        self.y = self.data["Y"]  # Output/Label
    
    # Perform baseline regression (baseline model: Linear Regression without feature scaling)
    def baseline_regression(self):
        model = LinearRegression()
        rmse_scores = np.sqrt(-cross_val_score(model, self.X, self.y, scoring="neg_mean_squared_error", cv=self.kf))
        y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)
        return rmse_scores, y_pred
    
    # Perform regression with feature scaling (model 1: Linear Regression with feature scaling)
    def feature_scaling_regression(self):
        model = make_pipeline(StandardScaler(), LinearRegression())
        rmse_scores = np.sqrt(-cross_val_score(model, self.X, self.y, scoring="neg_mean_squared_error", cv=self.kf))
        y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)
        return rmse_scores, y_pred
    
    # Perform decision tree regression (model 2: Decision Tree Regression)
    def decision_tree_regression(self):
        model = DecisionTreeRegressor()
        rmse_scores = np.sqrt(-cross_val_score(model, self.X, self.y, scoring="neg_mean_squared_error", cv=self.kf))
        y_pred = cross_val_predict(model, self.X, self.y, cv=self.kf)
        return rmse_scores, y_pred
    
    # Calculate average RMSE and standard deviation
    def calculate_rmse_stats(self, rmse):
        return {
            "Average RMSE": rmse.mean(),
            "Standard Deviation": rmse.std()
        }
    
    # Plot scatter plot
    def plot_scatter(self, y_true, y_pred, title, color):
        plt.scatter(y_true, y_pred, color=color)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.title(title)
        plt.xlabel('Actual Output')
        plt.ylabel('Predicted Output')
    
    # Visualize results
    def visualize_results(self):
        baseline_rmse, y_pred_baseline = self.baseline_regression()
        model1_rmse, y_pred_model_1 = self.feature_scaling_regression()
        model2_rmse, y_pred_model_2 = self.decision_tree_regression()

        results_rmse_avg = {
            "Baseline": self.calculate_rmse_stats(baseline_rmse),
            "Feature Scaling + Linear Regression": self.calculate_rmse_stats(model1_rmse),
            "Decision Tree Regression": self.calculate_rmse_stats(model2_rmse)
        }
        
        # Create a DataFrame from results
        results_df = pd.DataFrame(results_rmse_avg)
        print("----------------------------------------------------------------------------------------------")
        print("Average RMSE and Standard Deviation for each model:")
        print("----------------------------------------------------------------------------------------------")
        print(results_df)
        print("----------------------------------------------------------------------------------------------")
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        self.plot_scatter(self.y, y_pred_baseline, 'Baseline Model', 'blue')
        
        plt.subplot(1, 3, 2)
        self.plot_scatter(self.y, y_pred_model_1, 'Feature Scaling + Linear Regression', 'green')
        
        plt.subplot(1, 3, 3)
        self.plot_scatter(self.y, y_pred_model_2, 'Decision Tree Regression', 'red')
        
        plt.tight_layout()
        plt.show()

# Instantiate RegressionMethodAnalysis class object and perform analysis
regression_analysis = RegressionMethodAnalysis("A2data.tsv")
regression_analysis.visualize_results()
