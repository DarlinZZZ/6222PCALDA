import numpy as np
import matplotlib.pyplot as plt
import os
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
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time  # 引入time模块

# 1. 加载MNIST数据集方法、
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


# 3. 数据集划分方法（7:3比例）
def preprocess_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# 4. 数据标准化
def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# 5. PCA降维方法
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


# 8. 计算KL散度重构损失（判断协方差矩阵是否奇异）
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


# 9. 线性分类器（Logistic Regression）
def linear_classifier(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# 10. 最近邻分类器（K-Nearest Neighbors）
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


# 11. 可视化原始图像和重构图像
def visualize_original_vs_reconstructed(X_original, X_reconstructed, index, img_shape=(28, 28)):
    original_image = X_original[index].reshape(img_shape)
    reconstructed_image = X_reconstructed[index].reshape(img_shape)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('原始图像')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('重构图像')

    plt.show()


# 12. 主函数
# 12. 主函数
def main(dataset, DR, classifier_type):
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

    euclidean_losses = []  # 存储欧式距离重构损失
    mahalanobis_losses = []  # 存储马氏距离重构损失
    dimensions = []  # 存储降维维度

    # 对于 Iris 数据集（dataset=1），依次从降维维度 4 到 1 进行降维与重构
    if dataset == 1 and DR == 'PCA':
        for m_dimention in range(4, 0, -1):
            print(f"\n-----PCA 降维至 {m_dimention} 维-----")
            dimensions.append(m_dimention)

            start_time = time.time()  # 开始计时

            # PCA降维与重构
            X_train_reduced, X_train_reconstructed = apply_lda(X_train_scaled, m_dimention)
            X_test_reduced, X_test_reconstructed = apply_lda(X_test_scaled, m_dimention)

            # 计算重构损失
            euclidean_loss = calculate_reconstruction_loss_euclidean(X_train_scaled, X_train_reconstructed)
            mahalanobis_loss = calculate_reconstruction_loss_mahalanobis(X_train_scaled, X_train_reconstructed)
            euclidean_losses.append(euclidean_loss)  # 记录欧式距离重构损失
            mahalanobis_losses.append(mahalanobis_loss if mahalanobis_loss is not None else 0)  # 记录马氏距离重构损失

            print(f"PCA降至{m_dimention}维时的欧式距离重构损失: {euclidean_loss}")
            if mahalanobis_loss is not None:
                print(f"PCA降至{m_dimention}维时的马氏距离重构损失: {mahalanobis_loss}")
            else:
                print("由于奇异矩阵，未计算马氏距离重构损失")

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

            end_time = time.time()  # 结束计时
            time_elapsed = end_time - start_time  # 计算耗时
            print(f"维度 {m_dimention} 的PCA降维和分类用时: {time_elapsed:.4f} 秒")

        # 对于 MNIST 数据集（dataset=2），依次从降维维度 784 开始，每次减少 28 维，直到 1 维
    if dataset == 2 and DR == 'PCA':
        for m_dimention in range(784, 0, -28):
            print(f"\n-----PCA 降维至 {m_dimention} 维-----")
            dimensions.append(m_dimention)

            start_time = time.time()  # 开始计时

            # PCA降维与重构
            X_train_reduced, X_train_reconstructed = apply_pca(X_train_scaled, m_dimention)
            X_test_reduced, X_test_reconstructed = apply_pca(X_test_scaled, m_dimention)

            # 计算重构损失
            euclidean_loss = calculate_reconstruction_loss_euclidean(X_train_scaled, X_train_reconstructed)
            # mahalanobis_loss = calculate_reconstruction_loss_mahalanobis(X_train_scaled, X_train_reconstructed)
            euclidean_losses.append(euclidean_loss)  # 记录欧式距离重构损失
            # mahalanobis_losses.append(mahalanobis_loss if mahalanobis_loss is not None else 0)  # 记录马氏距离重构损失

            print(f"PCA降至 {m_dimention} 维时的欧式距离重构损失: {euclidean_loss}")
            # if mahalanobis_loss is not None:
            #     print(f"PCA降至 {m_dimention} 维时的马氏距离重构损失: {mahalanobis_loss}")
            # else:
            #     print("由于奇异矩阵，未计算马氏距离重构损失")

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

            end_time = time.time()  # 结束计时
            time_elapsed = end_time - start_time  # 计算耗时
            print(f"维度 {m_dimention} 的PCA降维和分类用时: {time_elapsed:.4f} 秒")

    if dataset == 2 and DR == 'LDA':
        for m_dimention in range(9, 0, -1):
            print(f"\n-----LDA 降维至 {m_dimention} 维-----")
            dimensions.append(m_dimention)

            start_time = time.time()  # 开始计时

            # LDA降维与重构
            X_train_reduced, X_test_reduced = apply_lda(X_train_scaled, y_train, X_test_scaled, m_dimention - 1)

            计算重构损失
            euclidean_loss = calculate_reconstruction_loss_euclidean(X_train_scaled, X_train_reconstructed)
            # mahalanobis_loss = calculate_reconstruction_loss_mahalanobis(X_train_scaled, X_train_reconstructed)
            euclidean_losses.append(euclidean_loss)  # 记录欧式距离重构损失
            mahalanobis_losses.append(mahalanobis_loss if mahalanobis_loss is not None else 0)  # 记录马氏距离重构损失

            print(f"PCA降至 {m_dimention} 维时的欧式距离重构损失: {euclidean_loss}")
            if mahalanobis_loss is not None:
                print(f"PCA降至 {m_dimention} 维时的马氏距离重构损失: {mahalanobis_loss}")
            else:
                print("由于奇异矩阵，未计算马氏距离重构损失")

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

            end_time = time.time()  # 结束计时
            time_elapsed = end_time - start_time  # 计算耗时
            print(f"维度 {m_dimention} 的PCA降维和分类用时: {time_elapsed:.4f} 秒")

    # 画折线图

    print(euclidean_losses)
    print(mahalanobis_losses)
    print(dimensions)

    plt.figure(figsize=(8, 6))
    plt.xticks(np.arange(min(dimensions), max(dimensions) + 1, 1))
    plt.plot(dimensions, euclidean_losses, color='red', label='Euclidean Distance')
    plt.plot(dimensions, mahalanobis_losses, color='blue', label='Mahalanobis Distance')

    plt.xlabel('m_dimention')
    plt.ylabel('recon loss')
    if dataset == 1:
        plt.title('Dataset = iris ' + '    DR method = ' + DR +'\nReconstruction Loss vs m_dimention')
    elif dataset == 2:
        plt.title('Dataset = MNIST ' + '    DR method = ' + DR + '\nReconstruction Loss vs m_dimention')
    else:
        plt.title('Dataset = CIFAR_10 ' + '    DR method = ' + DR + '\nReconstruction Loss vs m_dimention')
    plt.legend()

    # 保存图像到源文件夹
    if dataset == 1:
        output_path = os.path.join(os.getcwd(), 'iris '+ DR +' reconstruction_loss.png')
    elif dataset == 2:
        output_path = os.path.join(os.getcwd(), 'MNIST ' + DR + ' reconstruction_loss.png')
    else:
        output_path = os.path.join(os.getcwd(), 'CIFAR ' + DR + ' reconstruction_loss.png')
    plt.savefig(output_path)
    print(f"折线图已保存至: {output_path}")

    plt.show()


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
    main(dataset=1, DR='PCA', classifier_type='Both')
    # main(dataset=2, DR='LDA', classifier_type='Both')
