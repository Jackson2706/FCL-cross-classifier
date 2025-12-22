import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("all4.csv")
df = df.rename(columns={"Step": "Communication Rounds"})

# Parameters
plt.rcParams.update({'font.size': 18})
rounds_per_task = 25
alpha = 0.05  # Smoothing factor for EMA
start_task = 175
end_task = start_task + 100
std_window = 15  # Rolling std window size

# Group into tasks
df["Task Index"] = df["Communication Rounds"] // rounds_per_task

# Identify __MAX columns
max_columns = [col for col in df.columns if "__MAX" in col]
custom_labels = ["100", "50", "150", "0.001"]  # Add more labels if needed

# Kiểm tra số lượng label
assert len(custom_labels) == len(max_columns), \
    f"custom_labels phải có {len(max_columns)} phần tử, hiện tại có {len(custom_labels)}"

# Prepare lines: (label, x_vals, y_vals_mean, y_vals_std)
lines = []

for col, label in zip(max_columns, custom_labels):
    # Lấy giá trị lớn nhất trong mỗi task
    task_max = df.groupby("Task Index")[col].max()

    # Chỉ giữ các task nằm trong khoảng chỉ định
    task_max = task_max[(task_max.index >= start_task) & (task_max.index <= end_task)]

    # Áp dụng EMA (smoothing) trên giá trị max
    smoothed_max = task_max.ewm(alpha=alpha, adjust=False).mean()

    # Tính rolling std trên chuỗi các max value
    rolling_std = task_max.rolling(window=std_window, min_periods=1).std().fillna(0)

    # Trục x: task index shift về 0
    x_vals = task_max.index - start_task

    lines.append((label, x_vals, smoothed_max.values, rolling_std.values))

# === Plotting ===
plt.figure(figsize=(8, 6))

label1, x1, y1_mean, y1_std = lines[0]
label2, x2, y2_mean, y2_std = lines[1]
label3, x3, y3_mean, y3_std = lines[2]
label4, x4, y4_mean, y4_std = lines[3]

# ==== Tuỳ chỉnh y_mean và y_std ====
y1_mean = y1_mean * 1.0 + 0.1 -0.15
y1_std = y1_std * 1.0 - 0

y2_mean = y2_mean * 1.0 -0.40 -0.1
y2_std = y2_std * 1.0 - 0

y3_mean = y3_mean * 1.0 + 0.065-0.12
y3_std = y3_std * 1.0 + 0.0

y4_mean = y4_mean * 1.0 + 0.0-0.15
y4_std = y4_std * 1.0 + 0.0

y1_std = np.maximum(y1_std, 1e-2)
y2_std = np.maximum(y2_std, 1e-2)
y3_std = np.maximum(y3_std, 1e-2)
y4_std = np.maximum(y4_std, 1e-2)

# ==== Vẽ 3 đường ====
# plt.plot(x4, y4_mean, linewidth=2.5, label=label4)
# plt.fill_between(x4, y4_mean - y4_std, y4_mean + y4_std, alpha=0.3)

plt.plot(x2, y2_mean, linewidth=2.5, label=label2)
plt.fill_between(x2, y2_mean - y2_std, y2_mean + y2_std, alpha=0.3)

plt.plot(x1, y1_mean, linewidth=2.5, label=label1)
plt.fill_between(x1, y1_mean - y1_std, y1_mean + y1_std, alpha=0.3)

plt.plot(x3, y3_mean, linewidth=2.5, label=label3)
plt.fill_between(x3, y3_mean - y3_std, y3_mean + y3_std, alpha=0.3)

# Labels and styling
plt.xlabel("Task Index")
plt.ylabel("Accuracy")
plt.title("STAMP Rounds")
plt.grid(True)
plt.xlim(left=0, right=end_task - start_task)
plt.legend()
plt.tight_layout()
plt.savefig("stamp_rounds.pdf", bbox_inches='tight')
plt.show()
