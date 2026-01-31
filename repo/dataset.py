import sklearn
import sklearn.feature_extraction
import sklearn.datasets
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import sklearn.externals
import sklearn.cluster

import scipy.stats
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def datasets_demo():
    #读取鸢尾花数据集
    iris = sklearn.datasets.load_iris()
    #print(iris)
    #获取数据集描述
    print(iris.DESCR)
    #获取特征标签
    print(iris.feature_names)
    #获取特征值
    #data是一个any array类型，类似于matlab的矩阵？是numpy库实现的
    print(iris.data.shape)

    #划分数据集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print(x_train.shape)
    #读取波士顿房价数据集
    sklearn.datasets.fetch_20newsgroups()

def dict_demo():
    #测试字典特征提取
    data = [{'city':'杭州', 'temp':100}, {'city':'上海', 'temp':80}, {'city':'杭州','temp':60}]
    #1.实例化转换器
    transformer = sklearn.feature_extraction.DictVectorizer()
    #2.调用fit_transform()方法
    data_new = transformer.fit_transform(data)
    print(data_new)
    #3.返回矩阵
    transformer = sklearn.feature_extraction.DictVectorizer(sparse=False)
    data_new = transformer.fit_transform(data)
    print(data_new)

def text_demo():
    #测试文本特征提取
    data = ["life is is short, i like python", "life is too long, i dislike python"]
    #实例化提取器
    transformer = sklearn.feature_extraction.text.CountVectorizer()
    #调用
    data_new = transformer.fit_transform(data)
    print(data_new)
    #以矩阵形式返回
    print(data_new.toarray())
    #特征名称
    print(transformer.get_feature_names_out())

def Chinese_demo():
    #提取中文文本,自动分词
    data = ['不知道要写点什么，就这样算了吧','有班上就不错了，工作已经够痛苦了，为什么还要我自己找','求求你给我个班上吧']
    #用jieba库分词,返回值是一个词语生成器
    for line in data:
        generator = jieba.cut(line)
        line = ' '.join(list(generator))
        print(line)
        transformer = sklearn.feature_extraction.text.CountVectorizer()
        data_new = transformer.fit_transform(data)
        print(data_new.toarray())

def tfidf_demo():
    #用TF-IDF指标提取文本信息
    data = ['不知道要写点什么，就这样算了吧','有班上就不错了，工作已经够痛苦了，为什么还要我自己找','求求你给我个班上吧']
    #用jieba库分词,返回值是一个词语生成器
    for line in data:
        generator = jieba.cut(line)
        line = ' '.join(list(generator))
        print(line)
        transformer = sklearn.feature_extraction.text.TfidfVectorizer()
        data_new = transformer.fit_transform(data)
        print(data_new.toarray())

def minmax_scaler_demo():
    #测试归一化方法
    #1.获取数据(用pandas库进行处理)
    data = pd.read_csv("data.txt")
    #提取前三列
    print(data)
    data.iloc[:, :3]

    #2.实例化一个转换器类
    mscaler = sklearn.preprocessing.MinMaxScaler()
    #3.调用方法进行转换
    mscaler.fit_transform(data)

def standard_scaler_demo():
    #测试标准化方法
    #1.获取数据(用pandas库进行处理)
    data = pd.read_csv("data.txt")
    #提取前三列
    print(data)
    data.iloc[:, :3]

    #2.实例化一个转换器类
    mscaler = sklearn.preprocessing.StandardScaler()
    #3.调用方法进行转换
    mscaler.fit_transform(data)

def variance_demo():
     #测试方差方法
    #1.获取数据(用pandas库进行处理)
    data = pd.read_csv("data.txt")
    #提取前三列
    print(data)
    data.iloc[:, :3]

    #2.实例化一个转换器类
    variance = sklearn.feature_selection.VarianceThreshold(0)
    #3.调用方法进行转换
    data_new = variance.fit_transform(data)
    print(data_new)

    #3.计算两个变量之间的相关系数
    scipy.stats.pearsonr(data['pe_ratio', 'pa_ration'])
    plt.figure(figsize=(20,8), dpi=100)
    plt.scatter(data[''], data[''])
    plt.show()

def PCA_demo():
    #测试PCA降维
    data = [[2,8,4,5], [6,3,0,5], [5,4,9,1]]
    #1.实例化一个转换器类
    transformer = sklearn.decomposition.PCA(n_components=2)

    #2.调用fit_transform
    data_new = transformer.fit_transform(data)
    print(data_new)

def KNN_demo():
    #测试KNN算法，对鸢尾花数据集进行分类
    #1.获取数据
    iris = sklearn.datasets.load_iris()

    #2.数据集划分
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, random_state=6)

    #3.特征工程（标准化）
    transfer = sklearn.preprocessing.StandardScaler()
    x_train = transfer.fit_transform(x_train)
    #注意：对测试集的标准化也需要用训练集的平均值和标准差（fit的参数）
    x_test = transfer.transform(x_test)

    #4.训练KNN预估器
    #knn默认采用明可夫斯基距离
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    #4.1加入网格搜索和交叉验证
    param_grid = {"n_neighbors":[1,3,5,7,9,11]}
    knn = sklearn.model_selection.GridSearchCV(knn, param_grid, cv=10)
    knn.fit(x_train, y_train)
    
    #5.模型评估
    #5.1方法一，比对真实值和预测值
    y_predict = knn.predict(x_test)
    print(y_predict)
    print(y_test == y_predict)
    #5.3计算准确率
    score = knn.score(x_test, y_test)
    #整体的计算结果
    print("准确率：", score)
    #查看网格搜索结果
    print("最佳参数：", knn.best_params_)
    #最佳结果是交叉验证过程中的最佳，数据集与整体不一样
    print("最佳结果:", knn.best_score_)
    print("最佳估计器", knn.best_estimator_)
    print("交叉验证结果", knn.cv_results_)

def bayes_demo():
    #1,获取数据
    news = sklearn.datasets.fetch_20newsgroups(subset="all")
    #2.划分数据集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(news.data, news.target)
    #3.特征工程
    transfer = sklearn.feature_extraction.text.TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #4.朴素贝叶斯算法做预估器
    estimator = sklearn.naive_bayes.MultinomialNB()
    estimator.fit(x_train, y_train)
    #5.模型评估
    #方法1 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测值:", y_predict)
    print("对比:", y_test == y_predict)
    #方法2 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率:", score)

def decision_tree_demo():
    #决策树测试
    #1.iris
    #用决策树模型对鸢尾花数据集进行分类
    #1.获取数据集
    iris = sklearn.datasets.load_iris()
    #2.划分数据集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, random_state=22)
    #3.特征工程（不用做标准化，因为对数据尺度不敏感）
    #4.训练模型
    estimator = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)
    #5.模型评估
    #方法1 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测值:", y_predict)
    print("对比:", y_test == y_predict)
    #方法2 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率:", score)
    #6.决策树过程可视化
    sklearn.tree.export_graphviz(estimator, out_file='tree.dot', feature_names=iris.feature_names)

def titanic_demo():
    #1.获取数据
    titanic = []
    path = "../../data/titanic/train.csv"
    titanic_train = pd.read_csv(path)
    titanic.append(titanic_train)
    path = "../../data/titanic/test.csv"
    titanic_test = pd.read_csv(path)
    titanic.append(titanic_test)
    
    titanic_data = pd.concat(titanic, axis=0)
    #print(titanic_data)
    #筛选特征值和目标值
    x = titanic_data[["Pclass", "Age", "Sex"]]
    y = titanic_data["Survived"]
    #2.数据处理（缺失值处理）
    #填补平均值
    x["Age"].fillna(x["Age"].mean(), inplace=True)
    y.fillna(0, inplace=True)
    #转换成字典
    x = x.to_dict(orient="records")

    #3.确定特征值，目标值
    #4.划分数据集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=22)
    #5.特征工程
    transfer = sklearn.feature_extraction.DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #随机森林进行预测
    estimator = sklearn.ensemble.RandomForestClassifier()
    #加入网格搜索与交叉验证
    param_dict = {"n_estimators" : [120, 200, 300, 500, 800, 1200], "max_depth":[5,8,15,20,30]}
    estimator = sklearn.model_selection.GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)

    #5.模型评估
    #方法1 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测值:", y_predict)
    print("对比:", y_test == y_predict)
    #方法2 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率:", score)
    #6.决策树过程可视化
    #sklearn.tree.export_graphviz(estimator, out_file='titanic_tree.dot', feature_names=transfer.get_feature_names_out())
'''
  
    #6.模型训练
    estimator = sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=8)
    estimator.fit(x_train, y_train)
    #5.模型评估
    #方法1 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("预测值:", y_predict)
    print("对比:", y_test == y_predict)
    #方法2 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率:", score)
    #6.决策树过程可视化
    sklearn.tree.export_graphviz(estimator, out_file='titanic_tree.dot', feature_names=transfer.get_feature_names_out())
'''

def boston_demo1():
    #通过线性回归预测波士顿房价
    #正规方程
    #1.获取数据集
    path = "../../data/boston/boston.csv"
    boston = pd.read_csv(path)
    #print(boston)
    #2.划分数据集
    x = boston[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']]
    y = boston['medv']
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,random_state=22)
    #3.特征工程(无量纲化处理)
    transfer = sklearn.preprocessing.StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #4.预估器
    estimator = sklearn.linear_model.LinearRegression()
    estimator.fit(x_train, y_train)
    #5.模型评估
    print('权重系数1：', estimator.coef_)
    print('偏置1:', estimator.intercept_)
    y_pred = estimator.predict(x_test)
    error = sklearn.metrics.mean_squared_error(y_test, y_pred)
    print('均方误差1:', error)

def boston_demo2():
    #通过线性回归预测波士顿房价
    #梯度下降
    #1.获取数据集
    path = "../../data/boston/boston.csv"
    boston = pd.read_csv(path)
    #print(boston)
    #2.划分数据集
    x = boston[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']]
    y = boston['medv']
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,random_state=22)
    #3.特征工程(无量纲化处理)
    transfer = sklearn.preprocessing.StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #4.预估器
    estimator = sklearn.linear_model.SGDRegressor(learning_rate="constant",eta0=0.001, max_iter=10000)
    estimator.fit(x_train, y_train)
    #5.模型评估
    print('权重系数2：', estimator.coef_)
    print('偏置2:', estimator.intercept_)
    y_pred = estimator.predict(x_test)
    error = sklearn.metrics.mean_squared_error(y_test, y_pred)
    print('均方误差2:', error)

def boston_demo3():
    #通过线性回归预测波士顿房价
    #岭回归
    #1.获取数据集
    path = "../../data/boston/boston.csv"
    boston = pd.read_csv(path)
    #print(boston)
    #2.划分数据集
    x = boston[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']]
    y = boston['medv']
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,random_state=22)
    #3.特征工程(无量纲化处理)
    transfer = sklearn.preprocessing.StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #4.预估器
    estimator = sklearn.linear_model.Ridge(alpha=0.5,max_iter=10000)
    estimator.fit(x_train, y_train)
    #5.模型评估
    print('权重系数3：', estimator.coef_)
    print('偏置3:', estimator.intercept_)
    y_pred = estimator.predict(x_test)
    error = sklearn.metrics.mean_squared_error(y_test, y_pred)
    print('均方误差3:', error)

def cancer_demo():
    #1.获取数据
    path = "../../data/cancer/breast-cancer-wisconsin.data"
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 
        'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
        'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    cancer = pd.read_csv(path, names=column_names)
    #print(cancer)
    #2.数据处理（处理缺失值，加上names字段）
    #缺失值处理
    #2.1将?替换为np.nan
    cancer = cancer.replace(to_replace="?", value=np.nan)
    #2.2缺失样本
    cancer.dropna(inplace=True)
    #print(cancer.isnull().any())
    #3.划分数据集
    #3.1筛选特征值和目标值
    x = cancer.iloc[:, 1:-1]
    y = cancer['Class']
    #print(x, y)
    #3.2划分
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=22)
    #4.特征工程（无量纲化处理）
    transfer = sklearn.preprocessing.StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    print(x_train)
    #5.创建逻辑回归预估器
    estimator = sklearn.linear_model.LogisticRegression()
    estimator.fit(x_train, y_train)
    #6.模型评估
    print('权重系数:', estimator.coef_)
    print('偏置:', estimator.intercept_)
    y_pred = estimator.predict(x_test)
    error = sklearn.metrics.mean_squared_error(y_test, y_pred)
    print('均方误差:', error)
    #7.评估参数
    report = sklearn.metrics.classification_report(y_test, y_pred, labels=[2,4], target_names=["良性", "恶性"])
    print(report)
    #8.计算ROC AUC参数
    #8.1将y_test的分类结果转换为0,1
    y_true = np.where(y_test>3, 1, 0)
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    print(auc)
    #9.导出模型

def read_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    #提取点坐标
    points = []
    for line in lines:
        #忽略注释
        if line.startswith("#"):
            continue
        #提取坐标
        values = line.split()
        x = float(values[0])
        y = float(values[1])
        z = float(values[2])
        intensity = float(values[3])
        cla = float(values[4])
        points.append([x,y,z,intensity