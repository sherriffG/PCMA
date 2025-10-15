import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def cluster_symbols(I, Q, n_clusters=4,symbol_len=16,offset=3):
    """
    Clusters the I/Q symbols using KMeans clustering.

    Parameters:
    - I: Array of I values.
    - Q: Array of Q values.
    - n_clusters: Number of clusters to form.

    Returns:
    - labels: Cluster labels for each symbol.
    """
    I_symbols = I[offset::symbol_len]
    Q_symbols = Q[offset::symbol_len]
    

    IQ_symbols = np.column_stack((I_symbols, Q_symbols))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(IQ_symbols)
    labels = kmeans.labels_
    
    return labels

def symbol_error_rate(I_true, Q_true, I_pred, Q_pred, n_clusters=4, symbol_len=16, offset=3):
    """
    Computes the symbol error rate (SER) between true and predicted I/Q symbols.

    Parameters:
    - I_true: True I values.
    - Q_true: True Q values.
    - I_pred: Predicted I values.
    - Q_pred: Predicted Q values.
    - n_clusters: Number of clusters for KMeans.
    - symbol_len: Length of each symbol.
    - offset: Offset for symbol extraction.

    Returns:
    - ser: Symbol error rate.
    """
    labels_true = cluster_symbols(I_true, Q_true, n_clusters, symbol_len, offset=3)
    labels_pred = cluster_symbols(I_pred, Q_pred, n_clusters, symbol_len, offset=1)
    
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = np.sum((labels_true == i) & (labels_pred == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    ser = 1 - np.sum(cost_matrix[row_ind, col_ind]) / len(labels_true)
    return ser

def cluster_centers(I, Q, n_clusters=4, symbol_len=16, offset=3):
    I_symbols = I[offset::symbol_len]
    Q_symbols = Q[offset::symbol_len]
    IQ_symbols = np.column_stack((I_symbols, Q_symbols))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(IQ_symbols)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return labels, centers, IQ_symbols

def assign_to_centers(IQ_pred, centers):
    # 计算每个预测点到各中心的距离，并分配标签
    dists = np.linalg.norm(IQ_pred[:, None, :] - centers[None, :, :], axis=2)
    labels_pred = np.argmin(dists, axis=1)
    return labels_pred

def symbol_error_rate_center(I_true, Q_true, I_pred, Q_pred, n_clusters=4, symbol_len=16, offset=3):
    # 获取真值标签和中心
    labels_true, centers, IQ_true = cluster_centers(I_true, Q_true, n_clusters, symbol_len, offset)
    # 取预测符号
    I_pred_symbols = I_pred[offset::symbol_len]
    Q_pred_symbols = Q_pred[offset::symbol_len]
    IQ_pred = np.column_stack((I_pred_symbols, Q_pred_symbols))
    # 用真值中心对预测符号分类

    labels_pred = assign_to_centers(IQ_pred, centers)
    # 计算SER
    ser = np.mean(labels_true != labels_pred)
    return ser
import matplotlib.pyplot as plt

def plot_IQ_with_centers(I_true, Q_true, I_pred, Q_pred, n_clusters=4, symbol_len=16, offset=3, save_path=None):
    """
    绘制IQ图，标注分类中心和预测符号
    """
    # 获取真值聚类中心
    _, centers, _ = cluster_centers(I_true, Q_true, n_clusters, symbol_len, offset=3)
    # 取预测符号
    I_pred_symbols = I_pred[offset::symbol_len]
    Q_pred_symbols = Q_pred[offset::symbol_len]
    IQ_pred = np.column_stack((I_pred_symbols, Q_pred_symbols))
    # 用真值中心对预测符号分类
    labels_pred = assign_to_centers(IQ_pred, centers)

    plt.figure(figsize=(7, 7))
    # 画预测符号点，按类别着色
    scatter = plt.scatter(IQ_pred[:, 0], IQ_pred[:, 1], c=labels_pred, cmap='tab10', s=30, alpha=0.7, label='Predicted Symbols')
    # 画分类中心
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=120, label='Centers')
    for idx, (x, y) in enumerate(centers):
        plt.text(x, y, f'C{idx}', color='red', fontsize=12, ha='right', va='bottom')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('IQ Plot with Centers and Predicted Symbols')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()