import pandas as pd
import matplotlib.pyplot as plt

# 选择文件名
file_name = int(input("请输入文件编号 (1: pca accuracy(784-1), 2: pca accuracy(50-1), 3: lda accuracy): "))

# 根据用户选择加载文件
if file_name == 1:
    file_path = "pca accuracy(784-1).xlsx"
elif file_name == 2:
    file_path = "pca accuracy(50-1).xlsx"
elif file_name == 3:
    file_path = "lda accuracy.xlsx"
else:
    raise ValueError("无效的文件编号！请输入1、2或3。")

# 读取 Excel 文件
data = pd.read_excel(file_path)

# 提取公共数据列
dimensions = data['m_dimension']
linear_accuracy = data['Accuracy_linear']
knn_accuracy = data['Accuracy_knn']

# 如果文件编号是 1 或 2，绘制重构误差图
if file_name in [1, 2]:
    if 'reconstruction loss' in data.columns:
        reconstruction_loss = data['reconstruction loss']

        # 图1：重构损失随维度变化
        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, reconstruction_loss, marker='^', color='r', label='Reconstruction Error')
        plt.title('Reconstruction Error vs Dimensions (PCA)')
        plt.xlabel('Dimensions')
        plt.ylabel('Reconstruction Error')
        plt.grid(True)
        plt.gca().invert_xaxis()  # 将X轴从高到低排列
        plt.legend()
        plt.show()
    else:
        print("警告：文件中缺少重构损失数据，跳过该图。")

# 图2：线性分类器与KNN分类器准确度
plt.figure(figsize=(10, 6))
plt.plot(dimensions, linear_accuracy, marker='o', color='b', label='Linear Classifier Accuracy')
plt.plot(dimensions, knn_accuracy, marker='s', color='g', label='KNN Classifier Accuracy')
plt.title('Accuracy vs Dimensions (PCA)' if file_name != 3 else 'Accuracy vs Dimensions (LDA)')
plt.xlabel('Dimensions')
plt.ylabel('Accuracy')
plt.grid(True)
plt.gca().invert_xaxis()  # 将X轴从高到低排列

# 文件编号为 2 或 3 时，设置横坐标的最小间隔为 1
if file_name in [2, 3]:
    plt.xticks(ticks=range(int(dimensions.min()), int(dimensions.max()) + 1))

plt.legend()
plt.show()

if 'time(s)' in data.columns:
    time = data['time(s)']

    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, time, marker='*', color='m', label='Time (s)', linestyle='-')
    plt.title('Time vs Dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.gca().invert_xaxis()  # 将X轴从高到低排列
    plt.legend()
    plt.show()
else:
    print("警告：文件中缺少时间数据，跳过时间图。")
