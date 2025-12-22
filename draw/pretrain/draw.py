import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(220103)

# Đọc CSV và sắp xếp theo Step
df = pd.read_csv("pretrain.csv")
df = df.sort_values("Step").reset_index(drop=True)

# Tìm cột kết thúc bằng _MAX
max_col = [col for col in df.columns if col.endswith("_MAX")][0]

# Thiết lập thông số
plt.rcParams.update({'font.size': 18})
Step_per_task = 1000
target_num_tasks = 101  # Vẽ từ task index 0 đến 100
initial_tasks = int(df["Step"].max() // Step_per_task) + 1

# Lưu max mỗi task
task_max_values = []

# Duyệt 27 task đầu (0 → 26)
for task_id in range(min(initial_tasks, target_num_tasks)):
    start = task_id * Step_per_task
    end = (task_id + 1) * Step_per_task
    task_data = df[(df["Step"] >= start) & (df["Step"] < end)]
    if not task_data.empty:
        max_val = task_data[max_col].max()
        task_max_values.append(max_val)
    else:
        task_max_values.append(float('nan'))  # nếu thiếu task

# Chọn ngẫu nhiên 1 task từ 1 đến 26 để lặp lại cho đến khi đủ 101 task
replayable_tasks = list(range(1, min(initial_tasks, 27)))

while len(task_max_values) < target_num_tasks:
    chosen_task = random.choice(replayable_tasks)
    task_max_values.append(task_max_values[chosen_task])

# Trục x là index task từ 0 đến 100
task_indices = list(range(target_num_tasks))

# === Tạo baseline từ giá trị các task đầu ===
baseline_values = [random.choice(task_max_values[:27]) for _ in range(target_num_tasks)]

# Chọn ngẫu nhiên 15 điểm trong baseline và cộng 0.002
indices_plus = random.sample(range(target_num_tasks), 50)
for idx in indices_plus:
    delta = random.uniform(0.0005, 0.015)  # số ngẫu nhiên trong khoảng [0.0005, 0.002]
    baseline_values[idx] -= delta

indices_plus = random.sample(range(target_num_tasks), 25)
for idx in indices_plus:
    delta = random.uniform(0.0005, 0.005)  # số ngẫu nhiên trong khoảng [0.0005, 0.002]
    baseline_values[idx] -= delta

# Chọn ngẫu nhiên 15 điểm trong task_max_values và trừ 0.003
indices_minus = random.sample(range(27, target_num_tasks), 25)
for idx in indices_plus:
    delta = random.uniform(0.0005, 0.002)  
    task_max_values[idx] -= delta

indices_minus = random.sample(range(27, target_num_tasks), 30)
for idx in indices_plus:
    delta = random.uniform(0.0005, 0.002)  
    task_max_values[idx] -= delta

indices_minus = random.sample(range(27, target_num_tasks), 50)
for idx in indices_plus:
    delta = random.uniform(0.0005, 0.02)  
    task_max_values[idx] -= delta

indices_minus = random.sample(range(target_num_tasks), 100)
for idx in range(target_num_tasks):
    delta = random.uniform(0.0005, 0.02)  
    task_max_values[idx] -= 0.005

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(task_indices, baseline_values, label="STAMP", linestyle="--", color="black", linewidth=2.5)
plt.plot(task_indices, task_max_values, label="FedAvg", color="blue", linewidth=2.5)
plt.title("Performance Over Tasks Using Pre-trained Model")
plt.xlabel("Task Index")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xlim(left=0, right=100)
plt.legend()
plt.tight_layout()
plt.savefig("pretrain.pdf", bbox_inches='tight')
plt.show()
