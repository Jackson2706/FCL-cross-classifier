import numpy as np

def generate_unique_permutations_flat(client_num=100, classes=10, duplicate=20, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    array = np.zeros((client_num, duplicate * classes), dtype=int)
    used_permutations = set()

    for i in range(client_num):
        while True:
            perm = tuple(np.random.permutation(classes))
            if perm not in used_permutations:
                used_permutations.add(perm)
                break
        
        # Lặp lại permutation 10 lần cho client i
        repeated_perm = np.tile(perm, duplicate)
        array[i] = repeated_perm
    
    return array

num_clients = 100
num_classes = 100

permuted_array = generate_unique_permutations_flat(client_num=num_clients, classes=num_classes)

np.save("/root/projects/FCL/dataset/class_order/class_order_cifar100_100clients.npy", permuted_array)

print(permuted_array.shape)  # (100, 10000)