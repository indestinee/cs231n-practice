from lib import *




def test_knn():
    n = 10
    train_x = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    train_y = np.array([0, 1, 2, 3])
    data = uniform(-1, 1, (n, 2))
    from lib import knn

    nn_bf = knn.nn_bf()
    nn_bf.train(train_x, train_y)
    result = nn_bf.predict(data)
    for i in range(n):
        print(data[i], result[i])
    
    knn_bf = knn.knn_bf(4)
    knn_bf.train(train_x, train_y)
    result = knn_bf.predict(data, 1)
    for i in range(n):
        print(data[i], result[i])




test_knn()
