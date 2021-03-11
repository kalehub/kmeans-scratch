# k-means clustering
# implementasi dari : http://studyshut.blogspot.com/2018/12/contoh-perhitungan-manual-menggunakan.html
import numpy as np
import matplotlib.pyplot as plt


def read_dataset(dataset_dir):
    return np.genfromtxt(dataset_dir, delimiter=';')

def init_cluster(centroid, data, k):
    all_euclidean = np.array([])
    # for this project only!
    for i in range(k):
        # membuat array dengan elemen yang berulang
        centro = np.array(np.tile(centroid[i], (data.shape[0],1)))
        # proses pengurangan
        subt_arr = abs(np.subtract(centro, data))
        squared_arr = np.square(subt_arr)
        euclidean_distance = np.sqrt(squared_arr.sum(axis=1))
        all_euclidean = np.append(all_euclidean, euclidean_distance)
        print(euclidean_distance)
    
    # all_euclidean = all_euclidean.reshape(15,1)
    final_eucl = np.ones((15,3)) 
    all_euclidean = np.reshape(all_euclidean,(3,-1))
    for _ in all_euclidean:
        temp = _.reshape(15,-1)
        final_eucl = np.hstack((final_eucl, temp))
    final_eucl = final_eucl[:,3:]
    print(f'jarak euclidean :\n {final_eucl}')
    # mengambil nilai index minimal menggunakan np.argmin
    euclidean_min = np.hstack((final_eucl, np.argmin(final_eucl, axis=1).reshape(15,-1)))
    data_with_cluster = np.hstack((data, np.argmin(final_eucl, axis=1).reshape(15,-1))) 
    print(f'data dengan cluster :\n {data_with_cluster}')

    # deklarasi array cluster baru
    new_cluster = np.zeros_like(centroid)
    # menentukan kluster baru
    for k in range(0, (data_with_cluster[0].shape[0]-1)):
        # print(data_with_cluster[np.where(data_with_cluster[:,-1] == k)])
        array_cluster = data_with_cluster[np.where(data_with_cluster[:,-1] == k)] 
        avg_cluster = np.divide(np.sum(array_cluster,axis=0), array_cluster.shape[0])
        new_cluster[k] = avg_cluster[:3]
    print(new_cluster)

def main():
    # membaca dataset
    dataset_dir = 'datas.csv'
    dataset = read_dataset(dataset_dir)
    number_of_cluster = 3

    # pilih centroid acak
    start_centroid = dataset[np.random.choice(dataset.shape[0],number_of_cluster, replace=False)]
    print('starting centroid : ')
    print(start_centroid)            

    # proses klastering menggunakan k-means clusterin
    init_cluster(start_centroid, dataset, number_of_cluster)
    



if __name__ == '__main__':
    main()
