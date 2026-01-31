import scipy.stats
import numpy as np
from collections import Counter
import math
import ot

def cubic_root_of_sum_of_cubes(emd):
    a = emd[0]
    b = emd[1]
    c = emd[2]
    sum_of_cubes = a**3 + b**3 + c**3
    result = np.cbrt(sum_of_cubes)  # numpy.cbrt 自动处理负数的立方根
    return result

#自定义距离函数，采用Wasserstein距离
def sample_distance(a,b):
    #wasserstein_distance(u_values, v_values, u_weights, v_weights)
    #计算wasserstein距离时，应该将分布的下标放在value处，将分布的权重放在weight处
    #大胆猜测一下就是x在前，y在后
    #分别统计一下H0和H1、H2的wasserstein距离
    #a = np.array(a)
    #b = np.array(b)
    emd = []
    '''
    for i in range(3):
        u = np.array(a[i])
        if len(u)==0:
            continue
        u0 = u[:, 0]
        u1 = u[:, 1]
        v = np.array(b[i])
        if len(v)==0:
            continue
        v0 = v[:, 0]
        v1 = v[:, 1]
        dist0 = scipy.stats.wasserstein_distance(u0, v0)
        dist1 = scipy.stats.wasserstein_distance(u1, v1)
        dist = np.sqrt(dist0**2 + dist1**2)
    '''
    for i in range(3):
        u = np.array(a[i])
        if len(u)==0:
            u = np.array([[0,0]])
            #continue
        n1 = u.shape[0]
        w1 = np.full(n1, 1/n1)

        v = np.array(b[i])
        if len(v)==0:
            v = np.array([[0,0]])
            #continue
        n2 = v.shape[0]
        w2 = np.full(n2, 1/n2)

        M = ot.dist(u, v, metric='euclidean')  # 欧几里得距离矩阵
        dist = ot.emd2(w1, w2, M)

        emd.append(dist)

    #print("\n计算距离：", a, '-', b, '\n')
    #u_values = a[:,0]
    #v_values = b[:,0]
    #u_weights = a[:,1]
    #v_weights = b[:,1]

    #u_weights[u_weights[:] == 0] = 0.001
    #v_weights[v_weights[:] == 0] = 0.001
    #emd = scipy.stats.wasserstein_distance(u_values, v_values)
    ##print(emd)
    #return np.sum(emd)
    return cubic_root_of_sum_of_cubes(emd)

class KNNClassifier:
    def __init__(self, k=3, range_list=[], grid_search_on=False, voting_rate=5, dis_thresh=1.75):
        self.k = k
        self.range_list = range_list
        self.grid_search_on = grid_search_on
        self.voting_rate = voting_rate
        self.X_train = None
        self.y_train = None
        self.dis_thresh = dis_thresh

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    #对外接口
    def predict(self, X):
        X = np.array(X)
        #predictions, avg_distances = [self._predict(x) for x in X]
        res = [self._predict(x) for x in X]
        #print(res[0])
        return [row[0] for row in res]

    #对外接口，返回置信度
    def predict_prob(self, X):
        X = np.array(X)
        #predictions, avg_distances = [self._predict(x) for x in X]
        res = [self._predict(x) for x in X]
        #print(res[0])
        return [row[0] for row in res], [row[1] for row in res]

    #内部调用
    def transfer(self, X):
        X = np.array(X)
        #predictions, avg_distances = [self._predict(x) for x in X]
        res = [self._predict(x) for x in X]
        #print(res[0])
        return [row[0] for row in res], [row[1] for row in res]
   
    def _predict(self, x):
        # 计算所有训练样本与输入点的距离
        distances = [sample_distance(x_train, x) for x_train in self.X_train]
        # 获取最近 k 个样本的标签
        if self.grid_search_on:
            predictions = {}
            avg_distances = {}
            # 返回出现频率最高的标签
            for range in self.range_list:
                k_indices = np.argsort(distances)[:range]
                k_nearest_distances = [distances[i] for i in k_indices]
                k_nearest_labels = [self.y_train[i] for i in k_indices]
                mode, count = Counter(k_nearest_labels).most_common(1)[0]
                #阈值判断
                if count < self.voting_rate:
                    mode = 3.0
                #统计平均的Wasserstein距离
                avg_dist =  np.mean(k_nearest_distances)
                #阈值判断
                if avg_dist > self.dis_thresh:
                    mode = 3.0

                #类别
                predictions[range] = mode
                #到K个近邻的平均Wasserstein距离
                avg_distances[range] = avg_dist

            return predictions, avg_distances
        else:
            # 返回出现频率最高的标签
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            mode, count = Counter(k_nearest_labels).most_common(1)[0]
            if count < self.voting_rate:
                mode = 3.0
            #return Counter(k_nearest_labels).most_common(1)[0][0]
            k_nearest_distances = [distances[i] for i in k_indices]
            dist = np.mean(k_nearest_distances)
            if dist > self.dis_thresh:
                mode = 3.0
            #计算平均距离
            return mode, dist
        #return predictions
        #return Counter(k_nearest_labels).most_common(1)[0][0]
    
    def _predict_prob(self, x):
        # 计算所有训练样本与输入点的距离
        distances = [sample_distance(x_train, x) for x_train in self.X_train]
        # 获取最近 k 个样本的标签
        if self.grid_search_on:
            predictions = {}
            probabilities = {}
            # 返回出现频率最高的标签
            for range in self.range_list:
                k_indices = np.argsort(distances)[:range]
                k_nearest_distances = [distances[i] for i in k_indices]
                k_nearest_labels = [self.y_train[i] for i in k_indices]
                mode, count = Counter(k_nearest_labels).most_common(1)[0]
                
                #置信度
                weights = 1 / (k_nearest_distances + 1e-5)
                weighted_counts = Counter()
                for label, weight in zip(k_nearest_labels, weights):
                    weighted_counts[label] += weight
                most_common_class, weighted_sum = weighted_counts.most_common(1)[0]
                if (weighted_sum<self.voting_rate) or (np.mean(k_nearest_distances)>self.dis_thresh):
                    most_common_class = 3.0
                total_weight = sum(weights)
                confidence = weighted_sum/total_weight

                predictions[range] = most_common_class
                probabilities[range] = count/self.k * 0.5 + confidence*0.5 
            return predictions, probabilities
        else:
            # 返回出现频率最高的标签
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            k_nearest_distances = [distances[i] for i in k_indices]
            mode, count = Counter(k_nearest_labels).most_common(1)[0]

            #置信度
            weights = 1 / (k_nearest_distances + 1e-5)
            weighted_counts = Counter()
            for label, weight in zip(k_nearest_labels, weights):
                weighted_counts[label] += weight
            most_common_class, weighted_sum = weighted_counts.most_common(1)[0]
            if (weighted_sum<self.voting_rate) or (np.mean(k_nearest_distances)>self.dis_thresh):
                most_common_class = 3.0
            total_weight = sum(weights)
            confidence = weighted_sum/total_weight
            probability = count/self.k * 0.5 + confidence*0.5

            return most_common_class, probability
        #return predictions
        #return Counter(k_nearest_labels).most_common(1)[0][0]

    def score(self, X, y):
        #统计一下每个分类判别时平均的wasserstein距离
        if self.grid_search_on:
            predictions, avg_distances = self.transfer(X)
            res_list = {}
            emd_dist = {1.0:[],
                        2.0:[],
                        4.0:[]}
            for range in self.range_list:
                pre_range = [pre[range] for pre in predictions]
                accuracy = np.mean(pre_range == y)
                #print("\n参数k : 准确率    ", range, " : ", accuracy)
                res_list[range] = accuracy
                #统计每一类的平均wasserstein距离
                avg_dist_range = [dist[range] for dist in avg_distances]
                avg_distance = np.mean(avg_dist_range)
                #print("参数k : 平均wasserstein距离    ", range, " : ", avg_distance)
                #统计每个分类的wassersstein距离
            return res_list
        else:
            predictions, avg_distances = self.transfer(X)
            accuracy = np.mean(predictions == y)
            avg_distance = np.mean(avg_distances)
            return accuracy

#交叉验证
#网格搜索
def GridSearch(x_train, y_train, x_test, y_test, n_neighbors_range):
    best_score = 0
    best_n_neighbors = None

    knn = KNNClassifier(range_list=n_neighbors_range, grid_search_on=True)
    knn.fit(x_train, y_train)

    #评估模型
    score = knn.score(x_test, y_test)
    for range in n_neighbors_range:
        value = score[range]
        if value > best_score:
            best_score = value
            best_n_neighbors = range

    #for n_neighbors in n_neighbors_range:
    #    # 创建并训练KNN模型
    #    knn = KNNClassifier(k=n_neighbors)
    #    knn.fit(x_train, y_train)
    #    
    #    # 评估模型
    #    score = knn.score(x_test, y_test)
    #    print(f"n_neighbors: {n_neighbors}, Validation Accuracy: {score:.4f}")
    #    
    #    # 更新最佳参数
    #    if score > best_score:
    #        best_score = score
    #        best_n_neighbors = n_neighbors

    #print("\n最佳n_neighbors参数为:", best_n_neighbors)
    #print("最佳准确率为:", best_score)

    return score

#if __name__ == '__main__':
#    X = [[[1, 2]], [[2, 3]], [[3, 3]], [[6, 5]], [[7, 8]], [[8, 8]]]
#    y = [0, 0, 0, 1, 1, 1]
#
#    knn = KNNClassifier(k=3)
#    knn.fit(X, y)
#
#    test_data = [[[5,5]]]
#    print(knn.predict(test_data))