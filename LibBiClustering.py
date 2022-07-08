import numpy as np
import math
from sklearn.preprocessing import RobustScaler
from scipy import stats
import random
# random.seed( 123 )
from tqdm import tqdm
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
import collections
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from Simulation import BlobSimulationData
from Simulation import ReadRealDataset
# from Simulation import SimulationData
from Simulation import Visulization
from Evaluation import ClusteringAnalysis
from Evaluation import PermanceEvaluation
from Evaluation import CompareClusteringAnalysis

class Biclustering():
    
    def __init__(self):
        self.coverge_ref_min = 0.1
        self.coverge_ref_min_step = 0.001
        self.outlier_conv = 0.5
        self.class_num_c_fold = 2
        self.class_num_c_times = 1
        self.class_num_c_times_2 = 1
        self.needIF = False
        ### True False
        self.show_figure = False

    def process(self,Mat,class_num):
        flagged_cells = []
        predict_outliers= []
        pred_noninfo_vars = []
        cut_flagged_cells = []
        
        original_Mat = Mat
        # Mat = RobustScaler().fit(Mat).transform(Mat)
        # Mat = stats.zscore(Mat, axis=1, nan_policy='omit')
        
        missing = ~np.isfinite(Mat)
        mu = np.nanmedian(Mat, 0, keepdims=1)
        Mat = np.where(missing, mu, Mat)
        
        if self.needIF:
            IsoF = IsolationForest()
            IsoF.fit(Mat)
            y_pred = IsoF.predict(Mat)
            pred_outliers = [i for i in range(len(y_pred)) if y_pred[i] == -1]
            predict_outliers.extend(pred_outliers)
            # print (predict_outliers)
            pred_nor = [i for i in range(len(y_pred)) if y_pred[i] == 1]
            cutout_Mat = [Mat[i] for i in range(len(Mat)) if y_pred[i] != -1]
        else:
            y_pred = [1] * len(Mat)
            pred_nor = [i for i in range(len(y_pred)) if y_pred[i] == 1]
            cutout_Mat = Mat
        
        # cutout_Mat = RobustScaler().fit(cutout_Mat).transform(cutout_Mat)
        cutout_Mat = stats.zscore(cutout_Mat, axis=1, nan_policy='omit')
        mu = np.nanmedian(cutout_Mat, 0, keepdims=1)
        # print ('mu',mu)
        
        # missing = ~np.isfinite(cutout_Mat)
        # mu = np.nanmedian(cutout_Mat, 0, keepdims=1)
        # cutout_Mat = np.where(missing, mu, cutout_Mat)
        count_times = 0
        count_times_2 = 0
        
        old_temp_flag_cells = []
        set_out_rows = {}
        
        class_num_c = int(self.class_num_c_fold*class_num)
        new_predict_outliers = []
        
        while True:
            ## SpectralBiclustering SpectralCoclustering
            model = SpectralBiclustering(n_clusters=class_num_c)
            model.fit(cutout_Mat)
            # print (true_outliers)
            # print('labels',model.row_labels_)
            # print('cols',model.column_labels_)
            
            pred_labels = model.row_labels_
            pred_cols = model.column_labels_
            
            outlyings = []
            coll_ = collections.Counter(pred_labels)
            coll_2 = collections.Counter(pred_cols)
            
            sub_meds = []
            sub_meds_dict = {}
            count_out = 0
            for clu_i in list(set(pred_labels)):
                acco_rows = [i for i in range(len(pred_labels)) if pred_labels[i] == clu_i]
                # print (acco_rows)
                acco_cols = [i for i in range(len(pred_cols)) if pred_cols[i] == clu_i]
                # print (acco_cols)
                sub_mat = [cutout_Mat[acco_rows,j] for j in range(len(pred_cols)) if j in acco_cols]
                sub_med = np.median(sub_mat)
                sub_meds_dict[clu_i] = sub_med
                sub_meds.append(sub_med)
                if coll_[clu_i] < self.coverge_ref_min * len(pred_labels):
                    count_out += 1
                    # print ('2',clu_i,sub_med)
                # else:
                    # print ('1',clu_i,sub_med)
            sorted_sub_meds = sorted(sub_meds)
            outlying_label_scope = sorted_sub_meds[:max(int(count_out/2)+1,1)]
            outlying_label_scope.extend([i for i in sorted_sub_meds[min(-int(count_out/2)-1,-1):] if i not in outlying_label_scope])
            
            outlying_labels = []
            for clu_i in list(set(pred_labels)):
                if coll_[clu_i] < self.coverge_ref_min * len(pred_labels) and sub_meds_dict[clu_i] in outlying_label_scope:
                    outlying_labels.append(clu_i)
            print ('outlying_labels',outlying_labels)
            
            label_col_dict = {}
            for clu_i in list(set(pred_labels)):
                if clu_i in  outlying_labels:
                # or coll_2[clu_i] < self.coverge_ref_min * len(pred_cols):
                    outlyings.append(clu_i)
                    red_cols = [i for i in range(len(pred_cols)) if pred_cols[i] == clu_i]
                    if len(red_cols) >= self.outlier_conv * Mat.shape[1]:
                        red_rows = [i for i in range(len(pred_labels)) if pred_labels[i] == clu_i]
                        for red_row in red_rows:
                            predict_outliers.append(pred_nor[red_row])
                            new_predict_outliers.append(red_row)
                            flagged_cells.extend([[pred_nor[red_row],j] for j in range(len(pred_cols)) if [pred_nor[red_row],j] not in flagged_cells])
                            for j in range(len(pred_cols)):
                                cutout_Mat[red_row,j] = mu[0][j]
                    else:
                        label_col_dict[clu_i] = red_cols
            # print ('outlyings',outlyings)
            print (label_col_dict)    
            temp_flag_cells = []
            if len(outlyings) == 0:
                # break
                count_times_2 += 1
            # for outlying in outlyings:
            
            
            
            
            for i_0,pred_label in (enumerate(tqdm(pred_labels))):
                i = pred_nor[i_0]
                if pred_label not in label_col_dict.keys():
                    continue
                for j in label_col_dict[pred_label]:
                # for j,pred_col in (enumerate((pred_cols))):
                #     if pred_col != pred_label:
                #         continue
                    # print ([i,j])
                    if [i,j] not in flagged_cells:
                        flagged_cells.append([i,j])
                    if [i_0,j] not in cut_flagged_cells:
                        cut_flagged_cells.append([i_0,j])
                    if [i_0,j] not in temp_flag_cells:
                        temp_flag_cells.append([i_0,j])
                    if i_0 not in set_out_rows.keys():
                        set_out_rows[i_0] = 0
                    # print (Mat[i,j])
                    cutout_Mat[i_0,j] = mu[0][j]
                        # print (Mat[i,j])
            
            # for clo_i in list(set(pred_cols)):
            #     temp_Mat = Mat[:,pred_cols == clo_i]
            #     print (temp_Mat)
            #     print (temp_Mat.shape)
            #     model.fit(temp_Mat)
            #     temp_pred_labels = model.row_labels_
            #     print (clo_i,'temp_pred_labels',temp_pred_labels)
                
            if np.all(old_temp_flag_cells == temp_flag_cells):
                count_times += 1
            old_temp_flag_cells = temp_flag_cells
            if class_num_c > class_num:
                class_num_c -= 1
            if count_times > self.class_num_c_times:
                break
            if count_times_2 > self.class_num_c_times_2:
                break
            self.coverge_ref_min -= self.coverge_ref_min_step
            
        print (pred_labels)
        
        if self.show_figure:
            # original_Mat Mat cutout_Mat
            fit_data = np.array(original_Mat)[np.argsort(model.row_labels_)]
            fit_data = fit_data[:, np.argsort(model.column_labels_)]

            plt.matshow(fit_data, cmap=plt.cm.Blues)
            plt.title("After biclustering; rearranged to show biclusters")

            plt.show()
        # exit()
        
        pred_labels = model.row_labels_
        
        
        # print (cut_flagged_cells)
        for cut_flagged_cell in cut_flagged_cells:
            set_out_rows[cut_flagged_cell[0]] += 1
        print (set_out_rows)  
        # exit()  
        
        for set_out_row in set_out_rows:
            # print ('3',set_out_row,set_out_rows[set_out_row])
            if set_out_rows[set_out_row] >= self.outlier_conv * Mat.shape[1]:
                predict_outliers.append(pred_nor[set_out_row])
                new_predict_outliers.append(set_out_row)
        pred_labels = [pred_labels[i] for i in range(len(pred_labels)) if i not in new_predict_outliers]    
        
        
        
        for j,pred_j in enumerate(pred_cols):
            if pred_j in model.row_labels_:
                continue
            pred_noninfo_vars.append(j)
         
        return pred_labels,predict_outliers,flagged_cells,pred_noninfo_vars
    

if __name__ == '__main__':       
    ### BlobSimulationData      SimulationData
    
    SimulationData = 1
    
    if SimulationData == 1:
        Mat,outlying_cells,true_outliers,true_noninfos,true_infos,cluster_list,class_num = BlobSimulationData() 
    else:
        Mat,outlying_cells,true_outliers,true_noninfos,true_infos,cluster_list,class_num = ReadRealDataset()
    Mat_0 = Mat
    # print ('nan:',np.any(np.isnan(Mat)))
    # print (np.where(np.isnan(Mat)))
    # print ('inf:',np.all(np.isinf(Mat)))
    # print (np.where(np.isinf(Mat)))
    # print (Mat_0)
    apply = Biclustering()
    pred_labels,predict_outliers,flagged_cells,pred_noninfo_vars = apply.process(Mat,class_num)  
    print ('predict_outliers:',len(predict_outliers),'\n',predict_outliers[:10])  
    print ('flagged_cells:',len(flagged_cells),'\n',flagged_cells[:10]) 
    print ('true_noninfos:',len(true_noninfos),'\n',true_noninfos[:10])
    if SimulationData == 1:  
        Precision_row,Recall_row,f1_score_row,Precision_cell,Recall_cell,f1_score_cell = PermanceEvaluation(true_outliers,predict_outliers,outlying_cells,flagged_cells,true_noninfos,pred_noninfo_vars,true_infos)
    RI, ARI = ClusteringAnalysis(pred_labels,class_num,predict_outliers,cluster_list)
    if SimulationData == 0:
        CompareClusteringAnalysis(pred_labels,class_num,predict_outliers,cluster_list,flagged_cells,Mat,Mat_0)