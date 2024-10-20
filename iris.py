import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据集
iris = datasets.load_iris()
print(iris)
X = iris.data
y = iris.target

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. 维度缩减方法选择
methods = {
    'PCA': PCA,
    'LDA': LDA
}

# 5. 设置不同的维度数
dims = range(1, X_train.shape[1] + 1)

# 6. 记录结果
results = {method: [] for method in methods}

for method_name, Method in methods.items():
    for dim in dims:
        if method_name == 'LDA' and dim > (len(np.unique(y_train)) - 1):
            # LDA的最大维度是类别数-1
            continue
        if method_name == 'PCA':
            reducer = Method(n_components=dim)
        else:
            reducer = Method(n_components=dim)
        X_train_reduced = reducer.fit_transform(X_train, y_train)
        X_test_reduced = reducer.transform(X_test)

        # 7. 分类
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_train_reduced, y_train)
        y_pred = classifier.predict(X_test_reduced)

        # 8. 评估
        acc = accuracy_score(y_test, y_pred)
        results[method_name].append(acc)
        print('accuracy score = '+ str(acc))

# 9. 绘制结果
plt.figure(figsize=(10, 6))
for method_name in methods:
    plt.plot(dims[:len(results[method_name])], results[method_name], marker='o', label=method_name)
plt.xlabel('维度数')
plt.ylabel('分类准确率')
plt.title('维度数与分类准确率的关系')
plt.legend()
plt.grid(True)
plt.show()
