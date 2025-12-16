import numpy as np

# Đường dẫn tới file
file_path = 'class_order_cifar10.npy'

# Đọc file .npy
data = np.load(file_path)

print(data[0])