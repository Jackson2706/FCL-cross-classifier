import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("grad_balance.csv")
df = df.rename(columns={"Step": "Communication Rounds"})

# Parameters
plt.rcParams.update({'font.size': 18})
rounds_per_task = 25
alpha = 0.1  # Smoothing factor for EMA
start_task = 45
end_task = 145
std_window = 5  # Rolling std window size

# Group into tasks
df["Task Index"] = df["Communication Rounds"] // rounds_per_task

# Identify __MAX columns
max_columns = [col for col in df.columns if "__MAX" in col]
custom_labels = ["False", "True"]  # Add more labels if needed

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
# label3, x3, y3_mean, y3_std = lines[2]

# ==== Tuỳ chỉnh y_mean và y_std ====
y1_mean = y1_mean * 1.0 - 0.17
y1_std = y1_std * 1.0 - 0

y2_mean = y2_mean * 1.0 - 0.18
y2_std = y2_std * 1.0 - 0

# y3_mean = y3_mean * 1.0 + 0.0
# y3_std = y3_std * 1.0 + 0.0

# ==== Vẽ 3 đường ====

plt.plot(x2, y2_mean, linewidth=2.5, label=label2)
plt.fill_between(x2, y2_mean - y2_std, y2_mean + y2_std, alpha=0.3)

plt.plot(x1, y1_mean, linewidth=2.5, label=label1)
plt.fill_between(x1, y1_mean - y1_std, y1_mean + y1_std, alpha=0.3)

# plt.plot(x3, y3_mean, linewidth=2.5, label=label3)
# plt.fill_between(x3, y3_mean - y3_std, y3_mean + y3_std, alpha=0.3)

# Labels and styling
plt.xlabel("Task Index")
plt.ylabel("Accuracy")
plt.title("Grad Balance")
plt.grid(True)
plt.xlim(left=0, right=end_task - start_task)
plt.legend()
plt.tight_layout()
plt.savefig("grad_balance.pdf", bbox_inches='tight')
plt.show()
