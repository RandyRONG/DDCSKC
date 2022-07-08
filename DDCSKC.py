
import numpy as np
import math
from scipy import stats
import seaborn as sns
import random
random.seed( 123 )
from tqdm import tqdm
import line_profiler as lp
import pysparcl
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from Simulation import BlobSimulationData
from Simulation import ReadRealDataset
from Simulation import SimulationData
from Simulation import Visulization
from Evaluation import PermanceEvaluation
from Evaluation import GetConfusion

class DDC():
    
    def __init__(self):
        self.robLocC = 3
        self.phi_b = 2.5
        self.Standardization_delta = 0.845
        ### 99% tolerance setting yielding the same cutoff c = 2.576 .
        # self.chi2_crtical_value = np.sqrt( stats.chi2.pdf(0.01, 1))
        self.chi2_crtical_value =  2.576
        self.robCorr_treshold = 0.5
        self.iterations = 1
        
    def process(self,Mat):
        
        flagged_cells = []
        predict_outliers= []
        noninfo_vars = []
        
        for iteration in range(self.iterations):
            print ('iteration: ',iteration+1)
            Standarded_Mat = self.Standardization(Mat,1)
            # profile = lp.LineProfiler(self.BivariateRelations)
            # profile.enable()
            pred_Mat,residual_Mat,noninfo_vars = self.BivariateRelations(Standarded_Mat,noninfo_vars)
            # profile.disable()
            # profile.print_stats()
            # exit()
            Mat,flagged_cells,predict_outliers,change_sign = self.FlaggingOutliers(Standarded_Mat,residual_Mat,flagged_cells,predict_outliers)
            if change_sign == 0:
                break
                
        cleaned_Mat = self.ReplaceNA(Mat,pred_Mat,predict_outliers)
        return cleaned_Mat,predict_outliers,flagged_cells,noninfo_vars
        
        
    def TransformMatrix(self,m):
        Standarded_Mat_tuple = list(zip(*m))
        Standarded_Mat = [list(i) for i in Standarded_Mat_tuple]
        return Standarded_Mat
    def PhiFunction(self,t,b):
        return min(t*t, b*b)
    def GetrobLoc(self,variable):
        m1 = np.median([i for i in variable if not math.isnan(i)])
        s1 = np.median([abs(i-m1) for i in variable if not math.isnan(i)])
        robLoc = np.sum([((1-(((y-m1)/s1)/self.robLocC)**2)**2)*y for y in variable if not math.isnan(y)]) / np.sum([((1-(((y-m1)/s1)/self.robLocC)**2)**2) for y in variable if not math.isnan(y)]) 
        return robLoc
    
    def GetrobScale(self,centered_variable):
        s2 = np.median([abs(i) for i in centered_variable if not math.isnan(i)])
        raw = np.mean([self.PhiFunction(y/s2,self.phi_b) for y in centered_variable if not math.isnan(y)])
        robScale = s2 * np.sqrt((1/self.Standardization_delta) * raw)
        return robScale
    
    def GetStandrizedValues(self,variable):
        robLoc = self.GetrobLoc(variable)
        centered_variable = [i-robLoc for i in variable]
        robScale = self.GetrobScale(centered_variable)
        standarded_variable = [(i-robLoc)/robScale for i in variable]
        return standarded_variable
        
    def Standardization(self,Mat,GetNA):
        # print (pd.DataFrame(Mat))
        Standarded_Mat = []
        for j in range(0,len(Mat[0])):
            variable = [i[j] for i in Mat]
            standarded_variable = self.GetStandrizedValues(variable)
            if GetNA == 1:
                standarded_variable = [i if abs(i) <= self.chi2_crtical_value else np.nan for i in standarded_variable]
            Standarded_Mat.append(standarded_variable)
        Standarded_Mat = self.TransformMatrix(Standarded_Mat)
        print (pd.DataFrame(Standarded_Mat))
        return Standarded_Mat

    def BivariateRelations(self,Mat,noninfo_vars):
        
        # profile = lp.LineProfiler(self.GetrobScale)
        # profile.enable()
        
        pred_Mat = []
        residual_Mat = []
        # record_corr_dict = {}
        ## tqdm
        ### np.nan 1
        rob_corr_Mat = [[np.nan]*len(Mat[0]) for i in range(len(Mat[0]))]
        rob_corr_rec = {}
        
        for j1 in tqdm(range(0,len(Mat[0]))):
            if j1 in noninfo_vars:
                pred_Mat.append([i[j1] for i in Mat])
                residual_Mat.append([0 for i in Mat])
                continue
            variable_1 = [i[j1] for i in Mat]
            predicted_values = []
            robCorrs = []
            rob_corr_Mat[j1][j1] = 1.0
            rob_corr_rec[j1] = 0
            
            for j2 in (range(0,len(Mat[0]))):
                if j1 == j2 or j2 in noninfo_vars:
                    continue
                variable_2 = [i[j2] for i in Mat]
                # robCorr_available = 0
                # for key_ in record_corr_dict.keys():
                #     if [j1,j2] == [int(i) for i in key_.split(' ')]:
                #         robCorr = record_corr_dict[' '.join([str(j1),str(j2)])]
                #         robCorr_available = 1
                #         break
                #     elif [j2,j1] == [int(i) for i in key_.split(' ')]:
                #         robCorr = record_corr_dict[' '.join([str(j2),str(j1)])]
                #         robCorr_available = 1
                #         break
                # if robCorr_available == 0:
                variable_com1 = [variable_1[i]+variable_2[i] for i in range(len(variable_1))]
                if len(list(set(variable_com1))) == 1 and math.isnan(variable_com1[0]):
                    continue
                variable_com2 = [variable_1[i]-variable_2[i] for i in range(len(variable_1))]
                robScale1 = self.GetrobScale(variable_com1)
                robScale2 = self.GetrobScale(variable_com2)
                robCorr = abs((robScale1 ** 2 - robScale2 ** 2) / 4)
                    # record_corr_dict[' '.join([str(j1),str(j2)])] = robCorr
                rob_corr_Mat[j1][j2] = robCorr
                if robCorr < self.robCorr_treshold:
                    continue
                robSlope,filter_variable_2 = self.GetrobSlope(variable_1,variable_2)
                predicted_values.append([robSlope*i for i in filter_variable_2])
                robCorrs.append(robCorr)
                rob_corr_rec[j1] += 1
            # print (j1,len(robCorrs))
            if len(robCorrs) == 0:
                noninfo_vars.append(j1)
            
            if len(predicted_values) == 0:
                pred_Mat.append([np.nan]*len(filter_variable_2))
                residual_Mat.append([np.nan]*len(filter_variable_2))
                continue
            pred_single_Mat = []
            totaltobCorrs = np.sum(robCorrs) 
            for sub_idx in range(len(predicted_values)):
                sub_vector = [i*robCorrs[sub_idx]/totaltobCorrs for i in predicted_values[sub_idx]]
                pred_single_Mat.append(sub_vector)
            weighted_predicted_values = []
            
            try:
                for i3 in range(len(pred_single_Mat[0])):
                    # print (i3)
                    weighted_predicted_value = np.mean([pred_single_Mat[j3][i3] for j3 in range(len(pred_single_Mat)) if not math.isnan(pred_single_Mat[j3][i3])])
                    weighted_predicted_values.append(weighted_predicted_value)
            except:
                print (variable_1)
                print (variable_2)
                print (robSlope)
                print (filter_variable_2)
                print ('*'*10)
                print (predicted_values)
                print (robCorrs)
                print ((pred_single_Mat))
                print ((pred_single_Mat[0]))
            deshrinkage_weighted_predicted_values =  self.Deshrinkage(weighted_predicted_values,variable_1) 
            ## deshrinkage_weighted_predicted_values weighted_predicted_values
            diff_vector = [variable_1[i] - deshrinkage_weighted_predicted_values[i]  for i in range(len(variable_1))]
            diff_vector_robScale = self.GetrobScale(diff_vector)
            residual_values = [i/diff_vector_robScale for i in diff_vector]
            pred_Mat.append(deshrinkage_weighted_predicted_values)
            residual_Mat.append(residual_values)
        
        # profile.disable()
        # profile.print_stats()
        # exit()
        # print (rob_corr_Mat)
        x_axis_labels = [' '.join([str(i),str(rob_corr_rec[i])]) if i not in noninfo_vars else str(i) + '   '  for i in range(0,len(Mat[0]))]
        sns.heatmap(rob_corr_Mat, cmap='Reds',xticklabels=x_axis_labels)
        plt.show()
        # exit()
        return self.TransformMatrix(pred_Mat),self.TransformMatrix(residual_Mat),noninfo_vars
    
    def GetrobSlope(self,variable_1,variable_2):
        variable_com3 = [variable_1[i]/variable_2[i] for i in range(len(variable_1))]
        variable_com3_filter = [i for i in variable_com3 if  not math.isnan(i)]
        slope_estimate = np.median(variable_com3_filter)
        raw_residual = [variable_1[i]-slope_estimate*variable_2[i] for i in range(len(variable_1))]
        raw_residual_robScale = self.GetrobScale(raw_residual)
        filter_variable_com3 = []
        filter_variable_2 = []
        for i in range(len(variable_com3)):
            if abs(raw_residual[i])>self.chi2_crtical_value*raw_residual_robScale:
                filter_variable_com3.append(np.nan)
                filter_variable_2.append(np.nan)
            else:
                filter_variable_com3.append(variable_com3[i])
                filter_variable_2.append(variable_2[i])
        robSlope = np.median([i for i in filter_variable_com3 if not math.isnan(i)])
        return robSlope,filter_variable_2

    def Deshrinkage(self,predicted_values,true_values):
        robSlope,filter_predicted_values = self.GetrobSlope(true_values,predicted_values)
        deshrinkage_predicted_values = [robSlope*i for i in filter_predicted_values]
        return deshrinkage_predicted_values
            
    def FlaggingOutliers(self,Mat,residual_Mat,flagged_cells,outliers):
        change_sign = 0
        for j in range(0,len(residual_Mat[0])):
            
            for i in range(0,len(residual_Mat)):
                if Mat[i][j] == np.nan:
                    flagged_cells.append([i,j])
                    change_sign = 1
                elif abs(residual_Mat[i][j]) > self.chi2_crtical_value and [i,j] not in flagged_cells:
                    # print ([i,j])
                    flagged_cells.append([i,j])
                    # print ('Mat[i][j]',Mat[i][j])
                    Mat[i][j] = np.nan
                    change_sign = 1
        T_values = []
        for i in range(0,len(residual_Mat)):
            # T_value = np.mean([stats.chi2.cdf(x ** 2,df=1) for x in residual_Mat[i] if not math.isnan(x)])
            T_value = np.mean([x for x in residual_Mat[i] if not math.isnan(x)])
            T_values.append(T_value)
        standarded_T_values = self.GetStandrizedValues(T_values)
        new_outliers = [i for i in range(len(standarded_T_values)) if abs(standarded_T_values[i])> self.chi2_crtical_value]
        new_outliers = [i for i in new_outliers if i not in outliers]
        outliers.extend(new_outliers)        
        return  Mat,flagged_cells,outliers,change_sign

    def ReplaceNA(self,Mat,pred_Mat,predict_outliers):
        cleaned_Mat = Mat.copy()
        for j in range(0,len(cleaned_Mat[0])):
            for i in range(0,len(cleaned_Mat)):
                if not math.isnan(cleaned_Mat[i][j]):
                    continue
                cleaned_Mat[i][j] = pred_Mat[i][j]
                
        if len(predict_outliers) > 0:
            # print (len(cleaned_Mat))
            cleaned_Mat_2 = []
            for i in range(0,len(cleaned_Mat)):
                if i in predict_outliers:
                    continue
                cleaned_Mat_2.append(cleaned_Mat[i])
            cleaned_Mat = cleaned_Mat_2
            # print (len(cleaned_Mat))
        for j in range(0,len(cleaned_Mat[0])):
            for i in range(0,len(cleaned_Mat)):
                if math.isnan(cleaned_Mat[i][j]):
                    cleaned_Mat[i][j] = 0.0
        return cleaned_Mat

def ClusteringAnalysis(X,class_num,predict_outliers,cluster_list,true_noninfos,pred_noninfo_vars):
    
    # X_scale = preprocessing.scale(X) 
    # min_max_scaler = preprocessing.MinMaxScaler()  
    # X_minMax = min_max_scaler.fit_transform(X) 
    
    # X = X_scale
    # print (pd.DataFrame(X))
    print (pd.DataFrame(X))
    perm = pysparcl.cluster.permute(X,k=class_num)
    result = pysparcl.cluster.kmeans(X, k=class_num, wbounds=perm['bestw'])[0]
    print (result)
    weights = result['ws']
    wbound = result['wbound']
    predict_noninfo = [i for i in range(len(weights)) if weights[i]<0.01]
    pred_noninfo_vars.extend([i for i in predict_noninfo if i not in pred_noninfo_vars])
    if len(true_noninfos) != 0 and len(pred_noninfo_vars) != 0:
        print ('non informative variables:')
        GetConfusion(true_noninfos,pred_noninfo_vars,1)
    predict_clusters = result['cs']
    reduced_cluster_list = [cluster_list[i] for i in range(len(cluster_list)) if i not in predict_outliers]
    print (reduced_cluster_list)
    print (predict_clusters)
    RI = metrics.adjusted_rand_score(reduced_cluster_list, predict_clusters)
    print (RI)
    

if __name__ == '__main__':
    
    ### BlobSimulationData      SimulationData
    SimulationData = 0
    
    if SimulationData == 1:
        Mat,outlying_cells,true_outliers,true_noninfos,true_infos,cluster_list,class_num = BlobSimulationData() 
    else:
        Mat,outlying_cells,true_outliers,true_noninfos,true_infos,cluster_list,class_num = ReadRealDataset()
    # Visulization(1,2,Mat,cluster_list)
    # Visulization(len(true_infos)+1,len(true_infos)+2,Mat,cluster_list)
    apply = DDC()
    cleaned_Mat,predict_outliers,flagged_cells,pred_noninfo_vars = apply.process(Mat)       
    PermanceEvaluation(true_outliers,predict_outliers,outlying_cells,flagged_cells,true_noninfos,pred_noninfo_vars,true_infos)
    ClusteringAnalysis(np.array(cleaned_Mat),class_num,predict_outliers,cluster_list,true_noninfos,pred_noninfo_vars)
            
        
        