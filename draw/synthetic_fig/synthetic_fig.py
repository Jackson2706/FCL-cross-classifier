import matplotlib.pyplot as plt
import numpy as np

# Trục x: các mức độ dị biệt dữ liệu
x = ['α=100', 'α=10', 'α=1.0', 'α=0.1']
x_ticks = np.arange(len(x))

data = {
    'FedAvg': [[52, 50, 44, 37],            [38, 36.5, 27, 18],   [82, 79.5, 75.2, 67],        [65.5, 60, 49, 38]],
    'FL+EWC': [[60, 58, 46, 38],            [43.5, 35.8, 33, 25], [85, 84, 81, 69],            [66, 62, 52.5, 45.6]],
    'GLFC':   [[56.5, 55, 43, 36.5],        [39, 37.2, 29, 20],     [84, 81, 78, 65],          [66.6, 61.5, 51.4, 40.1]],
    'FedCIL': [[58, 56.5, 47.5, 38],        [43, 40.7, 31, 23],   [86, 82, 80, 68],            [67, 61.5, 49, 41]],
    'LANDER': [[59.4, 57.4, 45.2 , 35.2],   [41.5, 38, 31, 23],     [85.4, 81.5, 77, 67.5],    [67.3, 62.8, 48.3, 37.3]],
    'TARGET': [[57.4, 54.4, 46.1, 36.2],    [43.1, 39.5, 29, 21],     [84.8, 83, 81, 68.1],    [65.8, 62.4, 49.8, 39.8]],
    'FedL2P': [[60.4, 58.9, 47.6, 37.7],    [44.5, 39, 32.5, 23.5],     [86, 81, 80, 68.8],    [68, 63, 51, 40]],
    'FedWeIT':[[62, 60, 48, 38.9],          [44, 42, 33, 26],     [87.5, 81.3, 80, 68.5],      [65.2, 62.3, 53, 42.2]],
    'AFFCL':  [[60, 57, 49.8, 39.5],        [42, 41, 34, 25],     [85.6, 84, 79, 69.8],        [65.8, 63, 52.8, 39.5]],
    'FedSSI': [[63, 60, 50.5, 42],          [45.8, 43, 36.5, 27],   [88, 87, 83, 72],            [68.5, 65.6, 54.5, 47.6]],
    'STAMP': [[64, 63, 51, 43],          [47.8, 43.5, 39.5, 28],   [89.5, 88, 84.5, 72.5],            [69.5, 67.3, 55.2, 49.6]],

}

colors = {
    'FedAvg': 'blue',
    'FL+EWC': 'brown',
    'AFFCL': 'yellowgreen',
    'GLFC': 'pink',
    'FedCIL': 'green',
    'LANDER': 'red',
    'FedWeIT': 'purple',
    'FedSSI': 'orange',
    'TARGET': 'gray',
    'STAMP': 'Black',
    'FedL2P': 'cyan',
}
linestyles = {
    'STAMP': 'dashdot'
}
markers = {
    'FedSSI': 'o',
    'FedAvg': 'o',
    'FL+EWC': 'o',
    'AFFCL': 'o',
    'GLFC': 'o',
    'FedCIL': 'o',
    'LANDER': 'o',
    'FedWeIT': 'o',
    'FedSSI': 'o',
    'TARGET': 'o',
    'STAMP': 'o',
    'FedL2P': 'o',
}

titles = ['CIFAR10 Dataset', 'CIFAR100 Dataset', 'Digit10 Dataset', 'Office31 Dataset']
y_lims = [(34.5, 65), (16, 50), (65, 90), (35, 73)]
x_lims = (0,3)

fig, axs = plt.subplots(1, 4, figsize=(14, 4), sharex=False)

for i, ax in enumerate(axs):
    for name, values in data.items():
        line, = ax.plot(x_ticks, values[i], label=name, color=colors.get(name, None),
                        linestyle=linestyles.get(name, '-'), marker=markers.get(name, ''))
    ax.set_title(titles[i])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x)
    ax.set_ylim(y_lims[i])
    ax.set_xlim(x_lims)
    ax.set_xlabel("Data Heterogeneity")
    if i == 0:
        ax.set_ylabel("Test Accuracy")
    ax.grid(True)

# Chỉnh legend ở trên giữa
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.0),
    ncol=11,
    fontsize=10,
    frameon=True,              # Hiển thị khung
    fancybox=True,             # Làm tròn góc khung
    shadow=True,              # Không đổ bóng
    framealpha=1.0,            # Độ trong suốt của khung (1.0 là đậm)
    edgecolor='gray',          # Màu viền khung
    facecolor='white'          # Màu nền khung
)
# Tiêu đề chính (suptitle) đặt ở dưới
fig.text(0.5, -0.12, "Figure 3. Performance w.r.t data heterogeneity $\\alpha$ for four datasets.", 
         ha='center', fontsize=14)

fig.tight_layout(rect=[0, 0, 1, 0.93])  # Chừa không gian phía trên cho legend
plt.show()
