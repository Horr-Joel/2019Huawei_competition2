import tensorflow as tf
import keras.backend as K
import numpy as np 

class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self,validation_data):
        self.x_val,self.y_val=validation_data
    def on_epoch_begin(self,epoch,logs={}):
        #添加roc_auc_val属性
        starttime=time.time()
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')
        if not ('costtime' in self.params['metrics']):
            self.params['metrics'].append('costtime')
        return starttime
    def on_epoch_end(self,epoch,logs={}):
        starttime=self.on_epoch_begin(epoch)
        nowtime=time.time()

        costtime=nowtime-starttime
        #costtime=time.time()
        #print(costtime)
        y_pre=model.predict(self.x_val)
        logs['roc_auc_val']=float('-inf')
        if(self.validation_data):
            logs['roc_auc_val']=roc_auc_score(self.y_val.flatten(),y_pre.flatten())
            logs['costtime']=costtime
        print('auc:{auc},costtime:{costtime}'.format(auc=logs.get('roc_auc_val'),costtime=logs.get('costtime')))

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
# AUC for a binary classifier
def auc(y_true, y_pred):
	ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
	pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
	pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
	binSizes = -(pfas[1:]-pfas[:-1])
	s = ptas*binSizes
	return K.sum(s, axis=0)

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
	y_pred = K.cast(y_pred >= threshold, 'float32')
	# N = total number of negative labels
	N = K.sum(1 - y_true)
	# FP = total number of false alerts, alerts from the negative class labels
	FP = K.sum(y_pred - y_pred * y_true)
	return FP/N

#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
	y_pred = K.cast(y_pred >= threshold, 'float32')
	# P = total number of positive labels
	P = K.sum(y_true)
	# TP = total number of correct alerts, alerts from the positive class labels
	TP = K.sum(y_pred * y_true)
	return TP/P

