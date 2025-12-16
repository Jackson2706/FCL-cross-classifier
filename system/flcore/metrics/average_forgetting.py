import numpy as np

def metric_average_forgetting(task, accuracy_matrix):
    result = 0
    F_list = []
    
    for i in range(task):
        acc_values = [accuracy_matrix[t][i] for t in range(task)]
        max_acc = max(acc_values)
        
        current_acc = accuracy_matrix[task][i]

        F_list.append(max_acc - current_acc)
    
    result = np.mean(F_list) if F_list else 0
    
    return result