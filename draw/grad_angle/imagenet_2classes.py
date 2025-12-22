import pandas as pd
import matplotlib.pyplot as plt

# Từ điển ánh xạ label → màu
colors = {
    'FedAvg': 'blue',
    'FL+EWC': 'pink',
    'FedDBE': 'orange',
    'GLFC': 'purple',
    'FedCIL': 'green',
    'LANDER': 'cyan',
    'FedL2P': 'red',
    'FedWeIT': 'yellowgreen',
    'AF-FCL': 'lightblue',
    'FedSSI': 'orange',
    'TARGET': 'gray',
    'STAMP': 'black',
}

# Custom labels bạn muốn vẽ
custom_labels = ['FedAvg', 'FedL2P', 'STAMP', 'FedSSI']

# Dữ liệu
df = pd.read_csv('imagenet_2classes_g.csv')
x = df.iloc[:, 0]
max_columns = [col for col in df.columns if col.endswith('__MAX')]

# Map label to columns theo thứ tự custom_labels
label_to_column = dict(zip(custom_labels, max_columns))

# Sắp xếp các label theo thứ tự trong `colors`
sorted_labels = [label for label in colors if label in custom_labels]

# Nếu có label không trong `colors`, thêm vào cuối
other_labels = [label for label in custom_labels if label not in colors]
final_labels = sorted_labels + other_labels

# Cấu hình smoothing
ema_span = 80
std_window = 5
plt.rcParams.update({'font.size': 18})

# Vẽ hình
plt.figure(figsize=(8, 6))
for label in final_labels:
    if label not in label_to_column:
        print(f"⚠️ Label '{label}' không khớp với bất kỳ cột dữ liệu nào.")
        continue

    col = label_to_column[label]
    color = colors.get(label, 'black')  # fallback nếu label không có màu

    ema = df[col].ewm(span=ema_span, adjust=False).mean()
    std = df[col].rolling(window=std_window, min_periods=1).std()

    linestyle = '--' if label == 'STAMP' else '-'  # Dùng nét đứt cho STAMP
    plt.plot(x, ema, label=label, linewidth=2.5, color=color, linestyle=linestyle)
    plt.fill_between(x, ema - std, ema + std, alpha=0.2, color=color)

# Giao diện
plt.xlabel('Communication Rounds')
plt.ylabel('Gradient Angle Cosine')
plt.title('ImageNet1K - 2 classes/task')
plt.legend()
plt.grid(True)
plt.xlim(left=0, right=300)
plt.ylim(bottom=0.01, top=0.065)
plt.tight_layout()
plt.savefig("ImageNet1K-2C.pdf", bbox_inches='tight')
plt.show()
