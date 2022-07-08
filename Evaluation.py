
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy import stats
import numpy as np
from scipy.special import comb

def rand_index_score(clusters, classes):
    if type(clusters[0]) != int:
        clu_dict = {}
        for cluster in set(clusters):
            clu_dict[cluster] = len(clu_dict.keys())
    clusters = [clu_dict[i] for i in clusters]
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def ClusteringAnalysis(pred_labels,class_num,predict_outliers,cluster_list):
    predict_clusters = pred_labels
    reduced_cluster_list = [cluster_list[i] for i in range(len(cluster_list)) if i not in predict_outliers]
    # print (reduced_cluster_list)
    print (predict_clusters[:10])
    # print (len(reduced_cluster_list))
    print (len(predict_clusters))
    RI = rand_index_score(reduced_cluster_list, predict_clusters)
    print ('RI',round(RI,4))
    ARI = metrics.adjusted_rand_score(reduced_cluster_list, predict_clusters)
    print ('ARI',round(ARI,4))
    return round(RI,4),round(ARI,4)
   
def CompareClusteringAnalysis(pred_labels,class_num,predict_outliers,cluster_list,flagged_cells,Mat,Mat_0):
    print ('-'*20)
    print ('orginal:')
    # scaled_Mat_0 = stats.zscore(Mat_0, axis=1, nan_policy='omit')
    kmeans = KMeans(n_clusters=class_num).fit(Mat)
    predict_clusters = kmeans.labels_
    reduced_cluster_list = [cluster_list[i] for i in range(len(cluster_list)) if i not in predict_outliers]
    reduced_predict_clusters = [predict_clusters[i] for i in range(len(predict_clusters)) if i not in predict_outliers]
    # print (reduced_cluster_list)
    print (predict_clusters[:10])
    # print (len(reduced_cluster_list))
    print (len(predict_clusters))
    RI = rand_index_score(cluster_list, predict_clusters)
    print ('RI',round(RI,4))
    ARI = metrics.adjusted_rand_score(cluster_list, predict_clusters)
    print ('ARI',round(ARI,4))
    
    RI = rand_index_score(reduced_cluster_list, reduced_predict_clusters)
    print ('reduced RI',round(RI,4))
    ARI = metrics.adjusted_rand_score(reduced_cluster_list, reduced_predict_clusters)
    print ('reduced ARI',round(ARI,4))
    
    print ('After RSB:')
    Mat = np.delete(Mat,predict_outliers, axis = 0)
    kmeans = KMeans(n_clusters=class_num).fit(Mat)
    predict_clusters = kmeans.labels_
    reduced_cluster_list = [cluster_list[i] for i in range(len(cluster_list)) if i not in predict_outliers]
    # print (reduced_cluster_list)
    print (predict_clusters[:10])
    # print (len(reduced_cluster_list))
    print (len(predict_clusters))
    RI = rand_index_score(reduced_cluster_list, predict_clusters)
    print ('RI',round(RI,4))
    ARI = metrics.adjusted_rand_score(reduced_cluster_list, predict_clusters)
    print ('ARI',round(ARI,4))
    
    
    
    
    
        
    
def PermanceEvaluation(true_outliers,predict_outliers,outlying_cells,flagged_cells,true_noninfos,pred_noninfo_vars,true_infos):
    print ('-'*20)
    if len(predict_outliers) != 0 and len(true_outliers) != 0:
        print ('outliers:')
        Precision_row,Recall_row,f1_score_row = GetConfusion(true_outliers,predict_outliers,1)
    else:
        Precision_row,Recall_row,f1_score_row = 0,0,0
    # flagged_cells = [i for i in flagged_cells if i[1] in true_infos]
    # outlying_cells = [i for i in outlying_cells if i[1] in true_infos]
    if len(flagged_cells) != 0 and len(outlying_cells) != 0:
        print ('outlying cells:')
        Precision_cell,Recall_cell,f1_score_cell = GetConfusion(outlying_cells,flagged_cells,0)
    else:
        Precision_cell,Recall_cell,f1_score_cell = 0,0,0
    if len(true_noninfos) != 0 and len(pred_noninfo_vars) != 0:
        print ('non informative variables:')
        Precision,Recall,f1_score = GetConfusion(true_noninfos,pred_noninfo_vars,1)
    
    return Precision_row,Recall_row,f1_score_row,Precision_cell,Recall_cell,f1_score_cell
    
def GetConfusion(true_objects,predicted_objects,printYN):
    if printYN == 1:
        print (true_objects)
        print (predicted_objects)
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(predicted_objects)):
        if predicted_objects[i] in true_objects:
            TP = TP + 1
        else:
            FP = FP + 1
    for i in range(len(true_objects)):
        if true_objects[i]  not in predicted_objects:
            FN = FN + 1
    Precision = TP / (TP+FP)
    Recall = TP / (TP+FN)
    if Precision+Recall != 0:
        f1_score = (2*Precision*Recall) / (Precision+Recall)
    else:
        f1_score = 0
    print  ('[TP,FP,FN]',[TP,FP,FN],round(Precision,4),round(Recall,4),round(f1_score,4))
    return round(Precision,4),round(Recall,4),round(f1_score,4)