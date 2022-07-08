
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


# random.seed( 123 )

def Visulization(x_ax,y_ax,data,classes):
    plt.scatter([i[x_ax] for i in data], [i[y_ax] for i in data], c=classes, label=classes)
    plt.show()

def ReadRealDataset():
    
    # class_num = 8
    ### GeneData  emotions  AllBooks_baseline_DTM_Labelled  SCADI test
    select_dataset = 'SCADI'
    class_num_dict = {
        'GeneData':5,
        'emotions':3,
        'AllBooks_baseline_DTM_Labelled':8,
        'SCADI':7,
        'test':3
    }
    
    input_file = 'E:/KUL/Stat/thesis/clustering/data/{}.csv'.format(select_dataset)
    
    class_num = class_num_dict[select_dataset]
    
    outlying_cells = []
    true_outliers = []
    true_noninfos = []
    true_infos = []
    
    df = pd.read_csv(input_file)
    cluster_list = list(df.iloc[:,0])
    df = df.drop(df.columns[[0]], axis=1)
    Mat = df.to_numpy()
    
    return Mat,outlying_cells,true_outliers,true_noninfos,true_infos,cluster_list,class_num

def BlobSimulationData():
    p_informative = 100
    p_noninformative = 200
    p = p_informative + p_noninformative
    class_num  = 3
    num_per_class = 50
    sample_num = class_num*num_per_class
    contamination_prob = 0.05
    nan_prob = 0.05
    outlier_prob = 0.05
    # 10,100
    centroid_range = [10,100]
    # 10.0
    cluster_std = 10.0
    
    outlying_cell_dist_yn = False
    outlying_cell_dist = [100,10]
    outlying_cell_range = [100,150]
    
    outlier_dist = [100,10]
    
    data = []
    # cluster_nums = []
    outlying_cells = []
    outliers = []
    # cluster_list = []
    true_infos = [int(i) for i in range(0,p_informative,1)]
    true_noninfos = [int(i) for i in range(p_informative,p,1)]
    
    
    blob_data,target=make_blobs(n_samples=sample_num, n_features=p_informative,centers=class_num, \
                           cluster_std=cluster_std, center_box=(centroid_range[0], centroid_range[1]), shuffle=True)
    non_info_data = np.random.randn(sample_num,p_noninformative)
    data = np.concatenate((blob_data,non_info_data),axis=1)
    

    for j in range(p):
        for k in range(sample_num):
            decision_prob = random.random()
            if decision_prob < contamination_prob:
                if outlying_cell_dist_yn:
                    outlying_cell_value = np.random.uniform(low=outlying_cell_range[0], high=outlying_cell_range[1],size=1, )
                else:
                    outlying_cell_value = random.normalvariate(outlying_cell_dist[0], outlying_cell_dist[1])
                data[k][j] = outlying_cell_value
                outlying_cells.append([k ,j])
            decision_prob = random.random()
            if decision_prob < nan_prob:
                data[k][j] = np.nan

    for  k in range(sample_num):
        decision_prob = random.random()
        if decision_prob < outlier_prob:
            data[k] =  [random.normalvariate(outlier_dist[0], outlier_dist[1]) for i in range(p)] 
            outliers.append(k)           
    
    # print (data)
    # print (target)
    # print (outlying_cells)
    # print (outliers)
    
    return data,outlying_cells,outliers,true_noninfos,true_infos,target,class_num

def SimulationData():
    p_informative = 100
    p_noninformative = 200
    p = p_informative + p_noninformative
    class_num  = 3
    num_per_class = 50
    num_per_class_var = 1
    contamination_prob = 0.05
    nan_prob = 0.05
    outlier_prob = 0.05
    centroid_range = [10,100]
    true_centroids = np.arange(centroid_range[0],centroid_range[1]+1,(centroid_range[1]-centroid_range[0])/(class_num-1))
    outlying_cell_range = [100,200]
    outlier_range = [100,200]
    
    data = []
    cluster_nums = []
    outlying_cells = []
    outliers = []
    cluster_list = []
    true_infos = [int(i) for i in range(0,p_informative,1)]
    true_noninfos = [int(i) for i in range(p_informative,p,1)]
    for i in range(class_num):
        ci_length = max(abs(int(random.normalvariate(num_per_class, num_per_class_var))),1)
        cluster_nums.append(ci_length)
        #### 0; diff
        true_centroid = true_centroids[i]
        samples = np.random.randn(ci_length,p_informative)+true_centroid
        non_info_data = np.random.randn(ci_length,p_noninformative)
        sub_data = np.concatenate((samples,non_info_data),axis=1)
        for j in range(p):
            for k in range(ci_length):
                decision_prob = random.random()
                if decision_prob < contamination_prob:
                    sub_data[k][j] = np.random.uniform(low=outlying_cell_range[0], high=outlying_cell_range[1],size=1, )
                    outlying_cells.append([k + len(data),j])
                decision_prob = random.random()
                if decision_prob < nan_prob:
                    sub_data[k][j] = np.nan

        for  k in range(ci_length):
            cluster_list.append(i)
            decision_prob = random.random()
            if decision_prob < outlier_prob:
                sub_data[k] =  [random.normalvariate(outlier_range[0], outlier_range[1]) for i in range(p)] 
                outliers.append(k + len(data))           
        data.extend(sub_data)
    
    # self.Visulization(1,2,data,cluster_list)
    # self.Visulization(p_informative+1,p_informative+2,data,cluster_list)
    
    return data,outlying_cells,outliers,true_noninfos,true_infos,cluster_list,class_num

def Visulization(x_ax,y_ax,data,classes):
    plt.scatter([i[x_ax] for i in data], [i[y_ax] for i in data], c=classes, label=classes)
    plt.show()
    
