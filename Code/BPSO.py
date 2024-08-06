# Import important library
import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
import matplotlib.pyplot as plt
import time

# Preepare the dataset and labels
d1 = scipy.io.loadmat("D:/frequency_band_after_filt/Class1_freqband_10_all.mat")
d2 = scipy.io.loadmat("D:/frequency_band_after_filt/Class2_freqband_10_all.mat")
d3 = scipy.io.loadmat("D:/frequency_band_after_filt/Class3_freqband_10_all.mat")
#d4 = scipy.io.loadmat("D:/frequency_band_after_filt/Class2_freqband_10_all.mat")

data1=d1['class1_feat']
data2=d2['class2_feat']
data3=d3['Class3_feat']
#data4=d4['class2_feat']

features1 = np.hstack([data1[matrix_index] for matrix_index in range(8)])
features2 = np.hstack([data2[matrix_index] for matrix_index in range(8)])
features3 = np.hstack([data3[matrix_index] for matrix_index in range(8)])
#features4 = np.hstack([data4[matrix_index] for matrix_index in range(8)])

print(features1.shape,features2.shape,features3.shape)

features=np.vstack((features1,features2,features3))
features.shape

#Label
a = data1.shape[1]
b = data2.shape[1]
c = data3.shape[1]
#d = data4.shape[1]
values = [0] * a + [1] * b + [2] * c 
labels = np.array(values)

print(labels.shape)

#Implement BPSO

# Initialize parameters
num_particles = 10
num_iterations = 10
num_features = 136
c1 = 2  # cognitive constant
c2 = 2  # social constant
r = 0.5  # threshold for sigmoid function

random.seed(42)

results = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness_function(selected_features):
    X = features[:, selected_features == 1]
    y = labels 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
    recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
    kappa_rf = cohen_kappa_score(y_test, y_pred_rf)
    
    print("Selected features: {}, X shape: {}".format(np.sum(selected_features), X.shape))
    
    return accuracy_rf, f1_rf, precision_rf, recall_rf, kappa_rf

for w in np.arange(0.1, 1.1, 0.1):
    print(f"Running PSO with inertia weight: {w:.1f}")
     
    # Initialize particles and velocities
    particles = np.zeros([num_particles, num_features])
    selected = np.random.randint(num_features, size=(num_particles, 100))
    for i in range(num_particles):
        k = np.unique(selected[i])[:20]
        particles[i, k] = 1
    
    velocities = np.zeros((num_particles, num_features))
    personal_best_positions = particles.copy()
    personal_best_scores = np.zeros(num_particles)
    global_best_position = np.zeros(num_features)
    global_best_score = float(0)
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_kappa = 0
    best_num_features = 0

    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}/{num_iterations}")
        start_time = time.time()
        
        for i in range(num_particles):
            accuracy, f1, precision, recall, kappa = fitness_function(particles[i])
            fitness = accuracy  # Using accuracy as the fitness value
            num_selected_features = np.sum(particles[i])
            print(f"    Particle {i}: Fitness = {fitness}, F1 = {f1}, Precision = {precision}, Recall = {recall}, Kappa = {kappa}")
            
            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = particles[i].copy()
            
            if fitness > global_best_score:
                global_best_score = fitness
                global_best_position = particles[i].copy()
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_kappa = kappa
                best_num_features = num_selected_features
        
        for i in range(num_particles):
            r1 = 0.5
            r2 = 0.5
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particles[i]) +
                             c2 * r2 * (global_best_position - particles[i]))
            
            sigmoid_velocities = sigmoid(velocities[i])
            particles[i] = np.where(r < sigmoid_velocities, 1, 0)
            if sum(particles[i]) < 5:
                k = np.random.randint(num_features, size=20)
                particles[i, k] = 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"  Time taken for iteration {iteration + 1}: {elapsed_time:.2f} seconds")

        results.append({
            'inertia_weight': w,
            'iteration': iteration + 1,
            'global_best_score': global_best_score,
            'global_best_position': global_best_position.copy(),
            'accuracy': accuracy,
            'f1_score': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'kappa': best_kappa,
            'time_taken': elapsed_time,
            'num_selected_features': best_num_features
        })


# Extract the final results
weights = []
global_best_scores = []
f1_scores = []
precisions = []
recalls = []
kappas = []
times = []
num_features_selected = []

for result in results:
    if result['iteration'] == num_iterations:
        weights.append(result['inertia_weight'])
        global_best_scores.append(result['global_best_score'])
        f1_scores.append(result['f1_score'])
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        kappas.append(result['kappa'])
        times.append(result['time_taken'])
        num_features_selected.append(result['num_selected_features'])

# Create a DataFrame with the results
df = pd.DataFrame({
    'Inertia Weight': weights,
    'Global Best Accuracy': global_best_scores,
    'F1 Score': f1_scores,
    'Precision': precisions,
    'Recall': recalls,
    'Kappa': kappas,
    'Time Taken': times,
    'Num Selected Features': num_features_selected
})

# Print and save the DataFrame
print(df)
df.to_csv("D:/PSO_results/pso_res_bands_BSP_10_classes.csv", index=False)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(df['Inertia Weight'], df['Global Best Accuracy'], marker='o', linestyle='-', color='y', 
         markerfacecolor='blue', markeredgecolor='blue')
plt.title('Global Best Accuracy vs Inertia Weight of stimulas class')
plt.xlabel('Inertia Weight')
plt.ylabel('Global Best Accuracy')
plt.grid(False)
plt.show()
