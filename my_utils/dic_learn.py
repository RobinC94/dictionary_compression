import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

fit_algorithm = 'lars'
transform_algorithm = 'omp'
transform_n_nonzero_coefs = 4
n_iter = 500

def comp_kernel(kernel, n_components=16):
    num = np.shape(kernel)[3]
    channel = np.shape(kernel)[2]
    filter_size = np.shape(kernel)[:2]
    #print np.shape(kernel)
    k=kernel*10

    k.shape = (filter_size[0]*filter_size[1]*channel, num)
    k=k.transpose()
    v, index, x, e = dict_learn(k, n_components=n_components)
    x = [ i/10.0 for  i in x ]

    print 'num: ', num, 'comp: ', n_components, 'error: ', e

    return v, index, x, e

def dict_learn(y, n_components=16, n_iter=n_iter):
    n = np.shape(y)[0]
    dico = MiniBatchDictionaryLearning(n_components=n_components,
                                       n_iter=n_iter,
                                       alpha=0.5,
                                       fit_algorithm=fit_algorithm,
                                       transform_algorithm=transform_algorithm,
                                       transform_n_nonzero_coefs=transform_n_nonzero_coefs)
    v = dico.fit(y).components_
    A, B = dico.inner_stats_
    #print(A)
    #print(B)
    x = dico.transform(y)
    #print("y:",y)
    #print("x,",x)
    #print("dic:",v)

    a = []
    nz = np.nonzero(x)
    #print(np.nonzero(x))
    #print(np.where(nz[0] == 1))
    #print(len(nz), n*2)
    index = zip(nz[0], nz[1])
    #print(index)
    for i in index:
        a.append(x[i[0], i[1]])

    res = np.dot(x,v)
    error = np.mean(np.linalg.norm(y-res, axis=1))/10.0
    #error = 0
    #print(res[0][:12])
    #print(y[:12,0])
    #print(error)
    #print(index)
    return v, index, a, error

if __name__ == '__main__':
    kernel = np.random.rand(3,3,16,32)
    dic, index, a, error = comp_kernel(kernel, n_components=8)

    print(np.shape(dic))
    print(np.shape(index))
    print(np.shape(a))
    print(index)
    print(error)
