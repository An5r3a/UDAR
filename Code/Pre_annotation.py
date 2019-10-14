
# coding: utf-8

# In[1]:


import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 



# In[44]:


# Read datasets
path = '/Users/andy42i/Documents/Documentos/Python/'
#First we read data from House A
    
houseA = 'aruba_removed'
file_name_A = path + str(houseA) + '.csv'
    
data_A = pd.read_csv(file_name_A ,sep=',',header=0)
#data_A = data_A.replace(to_replace=r'A', value='', regex=True)
   
#Then we read data from House B and transform it in terms of House A
houseB = 'r2_removed'
simAB = 'aruba_ws_r2_sensor_similarity'


file_name_B = path + str(houseB) + '.csv'
file_name_sim = path + str(simAB) + '.txt'

data_B = pd.read_csv(file_name_B ,sep=',',header=0)
sim_AB = pd.read_csv(file_name_sim ,sep='\t',header=None)

#data_B = data_B.replace(to_replace=r'A', value='', regex=True)


#Final datasets
XA = data_A.iloc[:,0:data_A.shape[1]-1]
yA = data_A.iloc[:,data_A.shape[1]-1].astype(float)

B_hat = np.dot(data_B.iloc[:,0:data_B.shape[1]-1],sim_AB.iloc[:,0:sim_AB.shape[1]-1].T)
XB = B_hat
yB = data_B.iloc[:,data_B.shape[1]-1].astype(float)


# In[39]:


# ==== DATA PROCESSING ==== #
# Read datasets
path = '/Users/andy42i/Documents/Documentos/Python/'
#First we read data from House A
    
houseA = '2'
file_name_A = path + str(houseA) + '.csv'
    
data_A = pd.read_csv(file_name_A ,sep=',',header=None)
data_A = data_A.replace(to_replace=r'C\.', value='', regex=True)
    
#Then we read data from House B and transform it in terms of House A
houseB = '1'
#simAB = 'AB_s_sim'
simAB = 'Sensor_Sim_BC'
#simAB = 'Sensor_Sim_AC'


file_name_B = path + str(houseB) + '.csv'
file_name_sim = path + str(simAB) + '.csv'

data_B = pd.read_csv(file_name_B ,sep=',',header=None)
sim_AB = pd.read_csv(file_name_sim ,sep='\t',header=None)

data_B = data_B.replace(to_replace=r'B\.', value='', regex=True)
#data_B.columns = data_A.columns

#Final datasets
XA = data_A.iloc[:,1:data_A.shape[1]]
yA = data_A.iloc[:,0].astype(float)

#sample_B = data_B.sample(int(data_B.shape[0]*.2)).reset_index()
B_hat = np.dot(data_B.iloc[:,1:data_B.shape[1]],sim_AB)
XB = B_hat
yB = data_B.iloc[:,0].astype(float)


# In[45]:


HA_train, HA_test, yA_train, yA_test = train_test_split(XA, yA, test_size=0.2, random_state= 0)


# In[46]:


# ==== FIRST ANNOTATION ==== #
Proba_Dist_train = pd.DataFrame()
House = 'A - R1'
#th = .8
#fscore = []
thres = [.5,.55,.6,.65,.7,.75,.8,.85]
exp = 3

for ne in range(exp):
    for th in thres:

        #classifiers
        ##KNN
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(5)
        predicted_knn = knn.fit(HA_train, yA_train)
        knn_pred_label = predicted_knn.predict_proba(B_hat)
        pred_label2 = predicted_knn.predict(B_hat)
        rs2 = recall_score(yB, pred_label2, average='micro') 
        ps2 = precision_score(yB, pred_label2, average='micro') 
        fscore.append(((House,'knn',th,(2*(rs2*ps2)/(rs2+ps2)))))
        #Proba_Dist_train = pd.concat([Proba_Dist_train, pd.DataFrame(knn_pred_label)], axis= 1)

        ##RF
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        predicted_rf = rf.fit(HA_train, yA_train)
        rf_pred_label = predicted_rf.predict_proba(B_hat)
        pred_label2 = predicted_rf.predict(B_hat)
        rs2 = recall_score(yB, pred_label2, average='micro') 
        ps2 = precision_score(yB, pred_label2, average='micro') 
        fscore.append(((House,'RF',th,(2*(rs2*ps2)/(rs2+ps2)))))

        #Proba_Dist_train = pd.concat([Proba_Dist_train, pd.DataFrame(rf_pred_label)], axis= 1)


        ##LogisticRegression
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        predicted_lr = lr.fit(HA_train, yA_train)
        lr_pred_label = predicted_lr.predict_proba(B_hat)
        pred_label2 = predicted_lr.predict(B_hat)
        rs2 = recall_score(yB, pred_label2, average='micro') 
        ps2 = precision_score(yB, pred_label2, average='micro') 
        fscore.append(((House,'LR',th,(2*(rs2*ps2)/(rs2+ps2)))))


        #Proba_Dist_train = pd.concat([Proba_Dist_train, pd.DataFrame(lr_pred_label)], axis= 1)


        ##SVC
        from sklearn.svm import SVC
        svc = SVC()
        parameters  = {'C':[.1,1],'gamma':[.1,1]}
        smv_ = svm.SVC(kernel='rbf', probability = True, class_weight = 'balanced')
        clf = GridSearchCV(smv_, parameters)
        predicted_data1 = clf.fit(HA_train, yA_train)
        svm_pred_label = predicted_data1.predict_proba(B_hat)
        pred_label2 = predicted_data1.predict(B_hat)
        rs2 = recall_score(yB, pred_label2, average='micro') 
        ps2 = precision_score(yB, pred_label2, average='micro') 
        fscore.append(((House,'SVM',th,(2*(rs2*ps2)/(rs2+ps2)))))


        #Proba_Dist_train = pd.concat([Proba_Dist_train, pd.DataFrame(svm_pred_label)], axis= 1)

        # ==== ENSEMBLE ==== #
        len_p = svm_pred_label.shape[1]
        ensemble_knn = []
        ensemble_rf = []
        ensemble_lr = []
        ensemble_svm = []
        ensemble = np.zeros((XB.shape[0],4))
        tresh = th

        for row in range(XB.shape[0]):
            row_max = knn_pred_label[row].max()
            if row_max >= tresh:
                id_max = np.argmax(knn_pred_label[row])
                #ensemble_knn.append(knn_pred_label[row][id_max])
                ensemble_knn.append(id_max)
                ensemble[row][0] = id_max
            else:
                ensemble_knn.append(-1)
                ensemble[row][0] = -1


        for row in range(XB.shape[0]):
            row_max = rf_pred_label[row].max()
            if row_max >= tresh:
                id_max = np.argmax(rf_pred_label[row])
                #ensemble_rf.append(rf_pred_label[row][id_max])
                ensemble_rf.append(id_max)
                ensemble[row][1] = id_max
            else:
                ensemble_rf.append(-1)
                ensemble[row][1] = -1


        for row in range(XB.shape[0]):
            row_max = lr_pred_label[row].max()
            if row_max >= tresh:
                id_max = np.argmax(lr_pred_label[row])
                #ensemble_lr.append(lr_pred_label[row][id_max])
                ensemble_lr.append(id_max)
                ensemble[row][2] = id_max
            else:
                ensemble_lr.append(-1)
                ensemble[row][2] = -1


        for row in range(XB.shape[0]):
            row_max = svm_pred_label[row].max()
            if row_max >= tresh:
                id_max = np.argmax(svm_pred_label[row])
                #ensemble_svm.append(svm_pred_label[row][id_max])
                ensemble_svm.append(id_max)
                ensemble[row][3] = id_max
            else:
                ensemble_svm.append(-1)
                ensemble[row][3] = -1

        ensemble_pred = np.zeros(XB.shape[0],dtype=int)

        for row in range(ensemble.shape[0]):
            lab_aux = ensemble[row]
            r_index = [i for i in range(lab_aux.shape[0]) if (lab_aux[i] == -1)]
            new_a = np.delete(lab_aux, r_index)
            new_unique = np.unique(new_a, return_counts = True)
            if len(new_unique[1]) > 1:
                l1 = new_unique[0][0]
                c1 = new_unique[1][0]
                c2 = new_unique[1][1]
                if c1 == c2:
                    ensemble_pred[row] = -1
                else:
                    ensemble_pred[row] = l1
            else:
                if len(new_unique[1]) == 0:
                    ensemble_pred[row] = -1
                else:
                    if new_unique[1][0] >= 2:
                        ensemble_pred[row] = new_unique[0][0]
                    else:
                        ensemble_pred[row] = -1

        rs2 = recall_score(yB, ensemble_pred, average='micro') 
        ps2 = precision_score(yB, ensemble_pred, average='micro') 
        fscore.append(((House,'SE', th, (2*(rs2*ps2)/(rs2+ps2)))))



# In[47]:


fscore


# In[48]:


pd.DataFrame(fscore).to_csv("UDAR_preannotation_candidates_1010_2.csv")

