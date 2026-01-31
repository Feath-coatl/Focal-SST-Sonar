import numpy
import sklearn
from sklearn.cluster import KMeans
import pandas as pd

#用kmeans方法对点云数据进行聚类
def cluster_kmeans(sonar):
    #1.读取数据
    data = read_frame_pts("../../data/sonar/test1/data.pts")
    sonar = pd.DataFrame(data, columns=["x", "y", "z", "intensity", "class"])
    #show_frame(sonar)
    #sonar = sonar.iloc[:,0:-2]
    #print(sonar)
    #2.预估器流程
    estimator = KMeans(n_clusters=3)
    estimator.fit(sonar)
    y_predict = estimator.predict(sonar)
    #3.模型评估
    #用轮廓系数进行评估
    score = sklearn.metrics.silhouette_score(sonar, y_predict)
    print(y_predict)
    print(score)
    #4.可视化
    labels = estimator.labels_
    labels = pd.DataFrame(labels, columns=['class'])
    show_data = sonar.copy()
    show_data['class'] = labels

    #show_frame(show_data)

#if __name__ == __main__:
#    cluster_kmeans()
    