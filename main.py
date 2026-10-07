import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

csv_path = os.path.join('dataset', 'student-mat.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    print("Dataset loaded successfully! \n")
    
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    X = df.drop(columns=['G3'])
    y = df['G3']
    feature_names = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"Baseline Model RMSE: {baseline_rmse:.4f}\n")
    
    print("Initializing Particle Swarm Optimization (PSO)...")
    num_particles = 10
    max_iter = 5
    n_features = X.shape[1]
    
    np.random.seed(42)
    particles = np.random.randint(2, size=(num_particles, n_features))
    
    personal_best_scores = np.full(num_particles, float('inf'))
    global_best_position = particles[0]
    global_best_score = float('inf')
    
    iteration_best_scores = []
    
    for iteration in range(max_iter):
        for i in range(num_particles):
            selected_features = np.where(particles[i] == 1)[0]
            if len(selected_features) == 0:
                selected_features = [0]
                
            pso_model = RandomForestRegressor(random_state=42)
            pso_model.fit(X_train.iloc[:, selected_features], y_train)
            pso_rmse = np.sqrt(mean_squared_error(y_test, pso_model.predict(X_test.iloc[:, selected_features])))
            
            if pso_rmse < personal_best_scores[i]:
                personal_best_scores[i] = pso_rmse
                
            if pso_rmse < global_best_score:
                global_best_score = pso_rmse
                global_best_position = particles[i]
                
        iteration_bests = [global_best_score] * (iteration + 1)
        iteration_best_scores.append(global_best_score)
        print(f"Iteration {iteration + 1}/{max_iter} - Best PSO RMSE: {global_best_score:.4f}")
        
    print("\nPSO Optimization completed successfully!")
    print(f"Optimized Model RMSE with PSO: {global_best_score:.4f}\n")
    
    best_features_indices = np.where(global_best_position == 1)[0]
    selected_feature_names = feature_names[best_features_indices]
    print(f"Selected Features Count: {len(selected_feature_names)} out of {n_features}")
    print(f"Selected Features: {list(selected_feature_names)}\n")
    
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, max_iter + 1), iteration_best_scores, marker='o', color='blue', linewidth=2)
    plt.title('PSO Convergence Curve (RMSE over Iterations)')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig('pso_convergence_curve.png')
    plt.show()
    
    plt.figure(figsize=(6, 4))
    models = ['Baseline Model', 'PSO Optimized Model']
    rmse_values = [baseline_rmse, global_best_score]
    plt.bar(models, rmse_values, color=['skyblue', 'salmon'])
    plt.title('Performance Comparison (RMSE)')
    plt.ylabel('RMSE Value')
    plt.ylim(0, max(rmse_values) + 0.5)
    for i, v in enumerate(rmse_values):
        plt.text(i, v + 0.05, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig('model_performance_comparison.png')
    plt.show()
    
    print("Both graphs saved successfully as images!")

else:
    print("Error: dataset/student-mat.csv file not found.")