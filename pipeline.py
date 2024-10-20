import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml, load_iris
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, det, LinAlgError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import IncrementalPCA

def load_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0  # 将数据标准化至0-1之间
    y = mnist.target.astype(int)
    return X, y

def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X = np.concatenate([X_train, X_test])  # 合并训练集和测试集
    X = X.astype('float32') / 255.0  # 将数据标准化至0-1之间
    X = X.reshape(X.shape[0], -1)  # 展平每个图像到1维向量 (3072)
    y = np.concatenate([y_train, y_test]).flatten()
    return X, y


def preprocess_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# def apply_pca(X_scaled, n_components, batch_size=800):
#     pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
#     X_reduced = pca.fit_transform(X_scaled)
#     X_reconstructed = pca.inverse_transform(X_reduced)
#     return X_reduced, X_reconstructed

def apply_pca(X_scaled, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)
    return X_reduced, X_reconstructed

# LDA降维方法
def apply_lda(X_train, y_train, X_test, n_components):
    lda = LDA(n_components=n_components)
    X_train_reduced = lda.fit_transform(X_train, y_train)
    X_test_reduced = lda.transform(X_test)
    return X_train_reduced, X_test_reduced


def calculate_reconstruction_loss_euclidean(X_original, X_reconstructed):
    loss = np.mean(np.sum((X_original - X_reconstructed) ** 2, axis=1))
    return loss


# 7. 计算马氏距离重构损失（判断协方差矩阵是否奇异）
def calculate_reconstruction_loss_mahalanobis(X_original, X_reconstructed):
    try:
        cov_matrix = np.cov(X_original, rowvar=False)  # 计算协方差矩阵
        if det(cov_matrix) == 0:  # 判断是否为奇异矩阵
            print("协方差矩阵是奇异的，跳过马氏距离计算")
            return None
        cov_inv = inv(cov_matrix)  # 协方差矩阵的逆
        loss = np.mean([mahalanobis(x_orig, x_rec, cov_inv) for x_orig, x_rec in zip(X_original, X_reconstructed)])
        return loss
    except LinAlgError:
        print("协方差矩阵计算出错，跳过马氏距离计算")
        return None


def calculate_reconstruction_loss_kl(X_original, X_reconstructed):
    try:
        mean_original = np.mean(X_original, axis=0)
        cov_original = np.cov(X_original, rowvar=False)

        mean_reconstructed = np.mean(X_reconstructed, axis=0)
        cov_reconstructed = np.cov(X_reconstructed, rowvar=False)

        if det(cov_original) == 0 or det(cov_reconstructed) == 0:
            print("协方差矩阵是奇异的，跳过KL散度计算")
            return None

        cov_inv_reconstructed = inv(cov_reconstructed)
        term1 = np.trace(cov_inv_reconstructed @ cov_original)
        term2 = (mean_reconstructed - mean_original).T @ cov_inv_reconstructed @ (mean_reconstructed - mean_original)
        term3 = np.log(np.linalg.det(cov_reconstructed) / np.linalg.det(cov_original))
        kl_divergence = 0.5 * (term1 + term2 - X_original.shape[1] + term3)

        return kl_divergence
    except LinAlgError:
        print("协方差矩阵计算出错，跳过KL散度计算")
        return None


def linear_classifier(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def nearest_neighbor_classifier(X_train, y_train, X_test, y_test, n_neighbors=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

'''
def mahalanobis_classifier(X_train, y_train, X_test):
    classes = np.unique(y_train)
    means = {}
    cov_inv = None

    # 计算每个类的均值和总协方差的逆矩阵
    for c in classes:
        means[c] = np.mean(X_train[y_train == c], axis=0)

    cov_matrix = np.cov(X_train, rowvar=False)

    # 判断协方差矩阵是否为奇异
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("协方差矩阵为奇异矩阵，无法计算马氏距离")
        return None

    y_pred = []
    for x in X_test:
        distances = []
        for c in classes:
            diff = x - means[c]
            dist = np.sqrt(diff.T @ cov_inv @ diff)  # 马氏距离
            distances.append(dist)
        y_pred.append(classes[np.argmin(distances)])  # 预测为距离最近的类
    return np.array(y_pred)

'''


import matplotlib.pyplot as plt

def visualize_original_vs_reconstructed(m_dimension, X_original, X_reconstructed, index, img_shape=(28, 28)):
    original_image = X_original[index].reshape(img_shape)
    reconstructed_image = X_reconstructed[index].reshape(img_shape)

    plt.figure(figsize=(8, 4))
    plt.suptitle(f'm_dimension: {m_dimension}')

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')

    plt.show()

def main(m_dimention, dataset,DR, classifier_type='linear'):
    # 加载数据集
    if dataset == 1:
        X, y = load_iris_data()
    elif dataset == 2:
        X, y = load_mnist_data()
        img_shape = (28, 28)
    else:
        X, y = load_cifar10_data()
        img_shape = (32, 32, 3)

    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 标准化数据
    X_train_scaled = standardize_data(X_train)
    X_test_scaled = standardize_data(X_test)

    # PCA降维与重构
    if DR == 'PCA':
        X_train_reduced, X_train_reconstructed = apply_pca(X_train_scaled, m_dimention)
        X_test_reduced, X_test_reconstructed = apply_pca(X_test_scaled, m_dimention)

        # 计算重构损失
        euclidean_loss = calculate_reconstruction_loss_euclidean(X_train_scaled, X_train_reconstructed)
        mahalanobis_loss = calculate_reconstruction_loss_mahalanobis(X_train_scaled, X_train_reconstructed)
        kl_loss = calculate_reconstruction_loss_kl(X_train_scaled, X_train_reconstructed)

        print(f"PCA降至{m_dimention}维时的欧式距离重构损失: {euclidean_loss}")

        if mahalanobis_loss is not None:
            print(f"PCA降至{m_dimention}维时的马氏距离重构损失: {mahalanobis_loss}")
        else:
            print("由于奇异矩阵，未计算马氏距离重构损失")

        if kl_loss is not None:
            print(f"PCA降至{m_dimention}维时的KL散度重构损失: {kl_loss}")
        else:
            print("由于奇异矩阵，未计算KL散度重构损失")

        # 可视化部分原始图像与重构图像
        if dataset == 2 and 3:
            index = np.random.randint(0, X_train_scaled.shape[0])
            fixed_index = 0
            visualize_original_vs_reconstructed(m_dimention, X_train_scaled, X_train_reconstructed, fixed_index, img_shape)
        else:
            print('no image for iris')

        # 选择分类器并计算准确度
        if classifier_type == 'linear':
            accuracy = linear_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            print(f"线性分类器的测试集准确度: {accuracy:.4f}")
        elif classifier_type == 'nearest':
            accuracy = nearest_neighbor_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            print(f"最近邻分类器的测试集准确度: {accuracy:.4f}")
        elif classifier_type == 'Both':
            accuracy_linear = linear_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            accuracy_nearest = nearest_neighbor_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            print(f"线性分类器的测试集准确度: {accuracy_linear:.4f}")
            print(f"最近邻分类器的测试集准确度: {accuracy_nearest:.4f}")
        else:
            print("无效的分类器类型")

    elif DR == 'LDA':
        X_train_reduced, X_test_reduced = apply_lda(X_train_scaled, y_train, X_test_scaled, m_dimention-1)
        # 选择分类器并计算准确度
        if classifier_type == 'linear':
            accuracy = linear_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            print(f"线性分类器的测试集准确度: {accuracy:.4f}")
        elif classifier_type == 'nearest':
            accuracy = nearest_neighbor_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            print(f"最近邻分类器的测试集准确度: {accuracy:.4f}")
        elif classifier_type == 'Both':
            accuracy_linear = linear_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            accuracy_nearest = nearest_neighbor_classifier(X_train_reduced, y_train, X_test_reduced, y_test)
            print(f"线性分类器的测试集准确度: {accuracy_linear:.4f}")
            print(f"最近邻分类器的测试集准确度: {accuracy_nearest:.4f}")
        else:
            print("无效的分类器类型")


# 运行主函数
if __name__ == "__main__":
    '''
    1. Iris 数据集
    类别数：3（分别为 Setosa, Versicolor, Virginica）
    特征数：4（分别为花萼长度、花萼宽度、花瓣长度、花瓣宽度）
    2. MNIST 数据集
    类别数：10（分别为数字 0 至 9）
    特征数：784（每个图像为 28x28 像素，28*28 = 784）
    3. CIFAR-10 数据集
    类别数：10（分别为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）
    特征数：3072（每个图像为 32x32 像素，RGB 三个通道，32323 = 3072）

    PCA 最大降维维度为 min(样本数−1,特征数)  
    LDA 最大降维维度为 类别数-1
    '''
    main(m_dimention=4, dataset=1, DR='LDA', classifier_type='Both')
    # main(m_dimention=10, dataset=2, DR='LDA', classifier_type='Both')
