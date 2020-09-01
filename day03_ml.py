from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor


def linear1():
    boston = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #预估器
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)
    #得出模型
    print("正规方程-权重系数为：\n ",estimator.coef_)
    print("正规方程-便直为：\n ",estimator.intercept_)

    return None


def linear2():
    #1 获取数据
    boston = load_boston()
    #2 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    #3标准化

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)
    # 得出模型
    print("梯度下降-权重系数为：\n ", estimator.coef_)
    print("梯度下降-便直为：\n ", estimator.intercept_)

    return None


if __name__=='__main__':
    linear1()
    linear2()
    print()