import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

fit_algorithm = 'cd'
transform_algorithm = 'omp'
transform_n_nonzero_coefs = 2
n_iter=50

def comp_kernel(kernel, n_components=16):
    num = np.shape(kernel)[3]
    channel = np.shape(kernel)[2]
    filter_size = np.shape(kernel)[:2]
    #print np.shape(kernel)

    k = kernel
    k.shape = (filter_size[0]*filter_size[1]*channel, num)
    k=k.transpose()
    v, index, a1, a2, e = dict_learn(k, n_components=n_components)

    '''
    for i in range(channel):
        k = kernel[:,:,i,:]
        k.shape = (filter_size[0]*filter_size[1], num)
        #print k.shape
        k=k.transpose()
        v, index, a1, a2, e = dict_learn(k, n_components=n_components)
        dic_list.append(v)
        index_list.append(index)
        a_list.append(a1)
        b_list.append(a2)
        avg_e += e
    '''

    print 'num: ', num, 'comp: ', n_components, 'error: ', e

    return v, index, a1, a2, e

def dict_learn(y, n_components=16, n_iter=n_iter):
    n = np.shape(y)[0]
    dico = MiniBatchDictionaryLearning(n_components=n_components,
                                       n_iter=n_iter,
                                       fit_algorithm=fit_algorithm,
                                       transform_algorithm=transform_algorithm,
                                       transform_n_nonzero_coefs=transform_n_nonzero_coefs)
    v = dico.fit(y).components_
    x = dico.transform(y)

    a1=[]
    a2=[]
    nz = np.nonzero(x)
    index = []
    #print(np.nonzero(x))
    #print(np.where(nz[0] == 1))
    #print(len(nz), n*2)
    for i in range(n):
        nz_i = np.where(nz[0] == i)[0]
        if nz_i.shape[0] == 2:
            l=[nz[1][nz_i[0]], nz[1][nz_i[1]]]
            a1.append(x[i,l[0]])
            a2.append(x[i,l[1]])
            index.append([i,l[0]])
            index.append([i,l[1]])
        else:
            if nz[1][nz_i[0]] == 0:
                l=[nz[1][nz_i[0]],1]
            else:
                l = [nz[1][nz_i[0]], 0]
            a1.append(x[i, l[0]])
            a2.append(0)
            index.append([i, l[0]])
            index.append([i,l[1]])

    #res = np.dot(x,v)
    #error = np.mean(np.linalg.norm(y-res, axis=1))
    error = 0
    #print(res)
    #print(error)
    #print(index)
    assert len(index) == n*2
    return v, index, a1, a2, error

if __name__ == '__main__':
    inin = range(3*3*16*32)
    kernel = np.array(inin)
    kernel.shape = (3,3,16,32)
    dic, index, a, b, error = comp_kernel(kernel, n_components=8)

    print(np.shape(dic))
    print(np.shape(index))
    print(np.shape(a))
    print(np.shape(b))
    print(error)
