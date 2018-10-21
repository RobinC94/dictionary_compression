import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

fit_algorithm = 'cd'
transform_algorithm = 'omp'
transform_n_nonzero_coefs = 2
n_iter=100

def comp_kernel(kernel, n_components=16):
    num = np.shape(kernel)[3]
    channel = np.shape(kernel)[2]
    dic_list = []
    index_list = []
    a_list = []
    b_list = []
    avg_e = 0

    for i in range(channel):
        k = kernel[:,:,i,:]
        k.shape = (9, num)
        k=k.transpose()
        v, index, a1, a2, e = dict_learn(k, n_components=n_components)
        dic_list.append(v)
        index_list.append(index)
        a_list.append(a1)
        b_list.append(a2)
        avg_e += e

    print 'num: ', num, 'comp: ', n_components, 'avg error: ', avg_e/num

    return dic_list, index_list, a_list, b_list

def dict_learn(y, n_components=16, n_iter=n_iter):
    n = np.shape(y)[0]
    dico = MiniBatchDictionaryLearning(n_components=n_components,
                                       n_iter=n_iter,
                                       fit_algorithm=fit_algorithm,
                                       transform_algorithm=transform_algorithm,
                                       transform_n_nonzero_coefs=transform_n_nonzero_coefs)
    v = dico.fit(y).components_
    x = dico.transform(y)

    index = []
    a1=[]
    a2=[]
    nz = np.nonzero(x)[1]
    for i in range(n):
        l=[nz[i*2], nz[i*2+1]]
        index.append(l)
        a1.append(x[i,l[0]])
        a2.append(x[i,l[1]])
    res = np.dot(x,v)
    error = np.mean(np.linalg.norm(y-res, axis=1))
    #print(res)
    #print(error)
    return v, index, a1, a2, error

if __name__ == '__main__':
    inin = range(3*3*16*32)
    kernel = np.array(inin)
    kernel.shape = (3,3,16,32)
    dic_list, index_list, a_list, b_list = comp_kernel(kernel, n_components=8)

    print(np.shape(dic_list))
    print(np.shape(index_list))
    print(np.shape(a_list))
    print(np.shape(b_list))
