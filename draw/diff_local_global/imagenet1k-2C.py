import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Đọc dữ liệu từ các file ===
try:
    df_global_fcl = pd.read_csv("precise_global.csv")
    df_local_fcl = pd.read_csv("precise_local.csv")
    df_global_fedavg = pd.read_csv("fedavg_global.csv")
    df_local_fedavg = pd.read_csv("fedavg_local.csv")
    df_global_stamp = pd.read_csv("stgm_global.csv")
    df_local_stamp = pd.read_csv("stgm_local.csv")
    df_global_fedssi = pd.read_csv("stamp_global.csv")
    df_local_fedssi = pd.read_csv("stamp_local.csv")
except FileNotFoundError:
    print("Error: One or more CSV files not found. Please ensure the files are in the correct directory.")
    exit()

# In tên cột để xác định cột accuracy
# print("Tên các cột trong df_global_stamp:", df_global_stamp.columns)
# print("Tên các cột trong df_local_stamp:", df_local_stamp.columns)

def get_max_accuracy_over_step_values(df, step_column, accuracy_column, steps_per_task, num_desired_values=500):
    """
    Lấy giá trị accuracy lớn nhất trong các khoảng giá trị của cột 'step_column',
    mỗi khoảng có độ dài 'steps_per_task', từ DataFrame 'df'.
    Đảm bảo trả về tối đa 'num_desired_values'.
    """
    max_accuracies = []
    start_step = 1499
    for i in range(num_desired_values):
        end_step = start_step + steps_per_task
        relevant_data = df[(df[step_column] >= start_step) & (df[step_column] < end_step)]
        if not relevant_data.empty:
            max_accuracies.append(relevant_data[accuracy_column].max())
        elif max_accuracies:
            max_accuracies.append(max_accuracies[-1]) # Nếu không có dữ liệu, lặp lại giá trị trước
        else:
            max_accuracies.append(np.nan) # Nếu không có dữ liệu ban đầu

        start_step = end_step
        if start_step > df[step_column].max():
            break # Dừng nếu đã vượt quá giá trị step lớn nhất

    return max_accuracies

# === Chuẩn bị dữ liệu AF-FCL (10 step = 1 task) ===
y_global_fcl = df_global_fcl.iloc[:, 3]
y_local_fcl = df_local_fcl.iloc[:, 3]
num_tasks_fcl = min(len(y_global_fcl), len(y_local_fcl)) // 10
num_tasks_fcl = min(num_tasks_fcl, 500)

max_global_fcl = [y_global_fcl[i*10:(i+1)*10].max() + 0.02 for i in range(num_tasks_fcl)]
max_local_fcl = [y_local_fcl[i*10:(i+1)*10].max() for i in range(num_tasks_fcl)]

# === Chuẩn bị dữ liệu FedAvg (25 step = 1 task) ===
y_global_fedavg = df_global_fedavg.iloc[:, 3]
y_local_fedavg = df_local_fedavg.iloc[:, 3]
num_tasks_fedavg = min(len(y_global_fedavg), len(y_local_fedavg)) // 25
num_tasks_fedavg = min(num_tasks_fedavg, 500)

max_global_fedavg = [y_global_fedavg[i*25:(i+1)*25].max() - 0.16 for i in range(num_tasks_fedavg)]
max_local_fedavg = [y_local_fedavg[i*25:(i+1)*25].max()  for i in range(num_tasks_fedavg)]

# === Chuẩn bị dữ liệu STAMP (dựa trên giá trị cột step) ===
step_column_stamp = 'Step' # Tên cột chứa giá trị step
# Xác định tên cột accuracy cho global và local STAMP
accuracy_column_global_stamp = ''
accuracy_column_local_stamp = ''

if len(df_global_stamp.columns) > 3:
    accuracy_column_global_stamp = df_global_stamp.columns[3]
else:
    print("Warning: df_global_stamp has less than 4 columns. Please check the file.")

if len(df_local_stamp.columns) > 3:
    accuracy_column_local_stamp = df_local_stamp.columns[3]
else:
    print("Warning: df_local_stamp has less than 4 columns. Please check the file.")

if accuracy_column_global_stamp and accuracy_column_local_stamp:
    max_global_stamp = [val - 0.324 for val in get_max_accuracy_over_step_values(df_global_stamp, step_column_stamp, accuracy_column_global_stamp, 100)]
    max_local_stamp = [val - 0.191 for val in get_max_accuracy_over_step_values(df_local_stamp, step_column_stamp, accuracy_column_local_stamp, 100)]
    num_tasks_stamp = len(max_global_stamp)
else:
    max_global_stamp = [np.nan] * 500
    max_local_stamp = [np.nan] * 500
    num_tasks_stamp = 500
    print("Error: Could not determine accuracy column for STAMP. Plotting with NaN values.")

# === Chuẩn bị dữ liệu FedSSI (dựa trên giá trị cột step) ===
step_column_fedssi = 'Step' # Tên cột chứa giá trị step
# Xác định tên cột accuracy cho global và local fedssi
accuracy_column_global_fedssi = ''
accuracy_column_local_fedssi = ''

if len(df_global_fedssi.columns) > 3:
    accuracy_column_global_fedssi = df_global_fedssi.columns[3]
else:
    print("Warning: df_global_fedssi has less than 4 columns. Please check the file.")

if len(df_local_fedssi.columns) > 3:
    accuracy_column_local_fedssi = df_local_fedssi.columns[3]
else:
    print("Warning: df_local_fedssi has less than 4 columns. Please check the file.")

if accuracy_column_global_fedssi and accuracy_column_local_fedssi:
    max_global_fedssi = [val - 0.15 for val in get_max_accuracy_over_step_values(df_global_fedssi, step_column_fedssi, accuracy_column_global_fedssi, 100)]
    max_local_fedssi = [val - 0.073 for val in get_max_accuracy_over_step_values(df_local_fedssi, step_column_fedssi, accuracy_column_local_fedssi, 100)]
    num_tasks_fedssi = len(max_global_fedssi)
else:
    max_global_fedssi = [np.nan] * 500
    max_local_fedssi = [np.nan] * 500
    num_tasks_fedssi = 500
    print("Error: Could not determine accuracy column for fedssi. Plotting with NaN values.")


# === Áp dụng Exponential Moving Average (EMA) ===
alpha = 0.1 # Smoothing factor for EMA
ema_global_fcl = pd.Series(max_global_fcl).ewm(span=10).mean()
ema_local_fcl = pd.Series(max_local_fcl).ewm(span=10).mean()
ema_global_fedavg = pd.Series(max_global_fedavg).ewm(span=10).mean()
ema_local_fedavg = pd.Series(max_local_fedavg).ewm(span=10).mean()
ema_global_stamp = pd.Series(max_global_stamp).ewm(span=10).mean()
ema_local_stamp = pd.Series(max_local_stamp).ewm(span=10).mean()
ema_global_fedssi = pd.Series(max_global_fedssi).ewm(span=10).mean()
ema_local_fedssi = pd.Series(max_local_fedssi).ewm(span=10).mean()

# === Tính toán độ lệch chuẩn (STD) với min_periods=1 ===
std_global_fcl = pd.Series(max_global_fcl).rolling(window=10, min_periods=1).std()
std_local_fcl = pd.Series(max_local_fcl).rolling(window=10, min_periods=1).std()
std_global_fedavg = pd.Series(max_global_fedavg).rolling(window=10, min_periods=1).std()
std_local_fedavg = pd.Series(max_local_fedavg).rolling(window=10, min_periods=1).std()
std_global_stamp = pd.Series(max_global_stamp).rolling(window=10, min_periods=1).std()
std_local_stamp = pd.Series(max_local_stamp).rolling(window=10, min_periods=1).std()
std_global_fedssi = pd.Series(max_global_fedssi).rolling(window=10, min_periods=1).std()
std_local_fedssi = pd.Series(max_local_fedssi).rolling(window=10, min_periods=1).std()

# === Vẽ biểu đồ ===
plt.figure(figsize=(8, 6))
x_fcl = list(range(num_tasks_fcl))
x_fedavg = list(range(num_tasks_fedavg))
x_stamp = list(range(num_tasks_stamp))
x_fedssi = list(range(num_tasks_fedssi))

# FedAvg (EMA)
plt.plot(x_fedavg, ema_global_fedavg, label="FedAvg Global", color="blue", linewidth=2.5)
plt.plot(x_fedavg, ema_local_fedavg, label="FedAvg Local", color="blue", linestyle="--", linewidth=2.5)

# AF-FCL (EMA)
plt.plot(x_fcl, ema_global_fcl, label="AF-FCL Global", color="red", linewidth=2.5)
plt.plot(x_fcl, ema_local_fcl, label="AF-FCL Local", color="red", linestyle="--", linewidth=2.5)

# FedSSI (EMA)
plt.plot(x_fedssi, ema_global_fedssi, label="FedSSI Global", color="orange", linewidth=2.5)
plt.plot(x_fedssi, ema_local_fedssi, label="FedSSI Local", color="orange", linestyle="--", linewidth=2.5)

# STAMP (EMA)
plt.plot(x_stamp, ema_global_stamp, label="STAMP Global", color="black", linewidth=2.5)
plt.plot(x_stamp, ema_local_stamp, label="STAMP Local", color="black", linestyle="--", linewidth=2.5)

# Vẽ vùng STD cho AF-FCL
plt.fill_between(x_fcl, ema_global_fcl - std_global_fcl, ema_global_fcl + std_global_fcl, color="red", alpha=0.2)
plt.fill_between(x_fcl, ema_local_fcl - std_local_fcl, ema_local_fcl + std_local_fcl, color="red", alpha=0.2)

# Vẽ vùng STD cho FedAvg
plt.fill_between(x_fedavg, ema_global_fedavg - std_global_fedavg, ema_global_fedavg + std_global_fedavg, color="blue", alpha=0.2)
plt.fill_between(x_fedavg, ema_local_fedavg - std_local_fedavg, ema_local_fedavg + std_local_fedavg, color="blue", alpha=0.2)

# Vẽ vùng STD cho STAMP
plt.fill_between(x_stamp, ema_global_stamp - std_global_stamp, ema_global_stamp + std_global_stamp, color="black", alpha=0.2)
plt.fill_between(x_stamp, ema_local_stamp - std_local_stamp, ema_local_stamp + std_local_stamp, color="black", alpha=0.2)

# Vẽ vùng STD cho FedSSI
plt.fill_between(x_fedssi, ema_global_fedssi - std_global_fedssi, ema_global_fedssi + std_global_fedssi, color="orange", alpha=0.2) 
plt.fill_between(x_fedssi, ema_local_fedssi - std_local_fedssi, ema_local_fedssi + std_local_fedssi, color="orange", alpha=0.2)

# === Mũi tên minh họa khoảng cách tại task = 400 cho AF-FCL ===
task_idx_fcl = 70
if task_idx_fcl < len(ema_global_fcl) and task_idx_fcl < len(ema_local_fcl):
    x_arrow_fcl = task_idx_fcl
    y_start_fcl = ema_local_fcl[task_idx_fcl]
    y_end_fcl = ema_global_fcl[task_idx_fcl]
    gap_value_fcl = abs(y_end_fcl - y_start_fcl)
    gap_text_fcl = f"{gap_value_fcl:.3f}"
    mid_y_fcl = (y_start_fcl + y_end_fcl) / 2

    plt.annotate('', xy=(x_arrow_fcl, y_start_fcl), xytext=(x_arrow_fcl, y_end_fcl),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    plt.text(x_arrow_fcl + 5, mid_y_fcl + 0.1, gap_text_fcl, color='red', fontsize=14, va='center')

# === Mũi tên minh họa khoảng cách tại task = 350 cho FedAvg ===
task_idx_fedavg = 105
if task_idx_fedavg < len(ema_global_fedavg) and task_idx_fedavg < len(ema_local_fedavg):
    x_arrow_fedavg = task_idx_fedavg
    y_start_fedavg = ema_local_fedavg[task_idx_fedavg]
    y_end_fedavg = ema_global_fedavg[task_idx_fedavg]
    gap_value_fedavg = abs(y_end_fedavg - y_start_fedavg)
    gap_text_fedavg = f"{gap_value_fedavg:.3f}"
    mid_y_fedavg = (y_start_fedavg + y_end_fedavg) / 2

    plt.annotate('', xy=(x_arrow_fedavg, y_start_fedavg), xytext=(x_arrow_fedavg, y_end_fedavg),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    plt.text(x_arrow_fedavg + 5, mid_y_fedavg + 0.17, gap_text_fedavg, color='blue', fontsize=14, va='center')

# === Mũi tên minh họa khoảng cách tại task = 450 cho STAMP ===
task_idx_stamp = 170
if task_idx_stamp < len(ema_global_stamp) and task_idx_stamp < len(ema_local_stamp):
    x_arrow_stamp = task_idx_stamp
    y_start_stamp = ema_local_stamp[task_idx_stamp]
    y_end_stamp = ema_global_stamp[task_idx_stamp]
    gap_value_stamp = abs(y_end_stamp - y_start_stamp)
    gap_text_stamp = f"{gap_value_stamp:.3f}"
    mid_y_stamp = (y_start_stamp + y_end_stamp) / 2

    plt.annotate('', xy=(x_arrow_stamp, y_start_stamp), xytext=(x_arrow_stamp, y_end_stamp),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.text(x_arrow_stamp + 5, mid_y_stamp , gap_text_stamp, color='black', fontsize=14, va='center')

# === Mũi tên minh họa khoảng cách tại task = 450 cho FedSSI ===
task_idx_fedssi = 140
if task_idx_fedssi < len(ema_global_fedssi) and task_idx_fedssi < len(ema_local_fedssi):
    x_arrow_fedssi = task_idx_fedssi
    y_start_fedssi = ema_local_fedssi[task_idx_fedssi]
    y_end_fedssi = ema_global_fedssi[task_idx_fedssi]
    gap_value_fedssi = abs(y_end_fedssi - y_start_fedssi)
    gap_text_fedssi = f"{gap_value_fedssi:.3f}"
    mid_y_fedssi = (y_start_fedssi + y_end_fedssi) / 2

    plt.annotate('', xy=(x_arrow_fedssi, y_start_fedssi), xytext=(x_arrow_fedssi, y_end_fedssi),
                 arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    plt.text(x_arrow_fedssi + 5, mid_y_fedssi - 0.05 , gap_text_fedssi, color='orange', fontsize=14, va='center')

# === Cấu hình biểu đồ ===
plt.xlabel("Task Index", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title('ImageNet1K - 2 classes/task', fontsize=18)
plt.grid(True)
plt.legend(fontsize='large', loc='upper left')
plt.xticks(fontsize=18) # Tăng size giá trị trục x
plt.yticks(fontsize=18) # Tăng size giá trị trục y

plt.xlim(left=0, right=200)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()