import pandas as pd 
import numpy as np
import config
import gc
import os
import pickle
from sklearn.preprocessing import LabelEncoder 
class DataLoader():
    def __init__(self,chunksize, **kwargs):
        self.chunksize = chunksize
        # has feature nunique dict and feature labelencoder dict？
        has_dict = False
        
        if os.path.exists('feature_dict.bin'):
            has_dict = True
            load_file = open("feature_dict.bin","rb")
            feature_dict = pickle.load(load_file)
            le_dict = pickle.load(load_file)
            load_file.close()
        else:
            feature_dict = {}
            le_dict = {}
            reader = pd.read_csv('../train_20190518.csv',header=None, chunksize=10000000)
            # ignore  operTime 
            for each in ['uId','adId','siteId','slotId','contentId','netType']:
                feature_dict[each] = set()
            for each in reader:
                each.columns = ['label','uId','adId','operTime','siteId','slotId','contentId','netType']
                each.fillna(-1,inplace=True)
                for f in ['uId','adId','siteId','slotId','contentId','netType']:
                    feature_dict[f] |= set(each[f].unique())
            

        self.ad_info = pd.read_csv('../ad_info.csv',header=None)
        self.ad_info.columns = ['adId','billId','primId','creativeType','intertype','spreadAppId']
        for f in ['adId','primId','creativeType','intertype']:
            self.ad_info[f] = self.ad_info[f].astype(np.int32)
        del self.ad_info['spreadAppId']
        self.ad_info.fillna(-1,inplace=True)

        self.user_info = pd.read_csv('../user_info.csv',header=None)
        self.user_info.columns = ['uId','age','gender','city','province','phoneType','carrier']
        for f in ['age','gender','city','province','phoneType','carrier']:
            self.user_info[f] = self.user_info[f].astype(np.float32)
        self.user_info.fillna(-1,inplace=True)

        self.content_info = pd.read_csv('../content_info.csv',header=None)
        self.content_info.columns=['contentId','firstClass','secondClass']
        self.content_info['contentId'] = self.content_info['contentId'].astype(np.int32)
        #content_info['secondClass'] = content_info['secondClass'].apply(lambda x: x.split('#') if isinstance(x,str) else [])
        self.content_info.fillna(-1,inplace=True)

        if not has_dict:
            for each in self.ad_info.columns:
                feature_dict[each] |= set(self.ad_info[each].unique())
            for each in self.user_info.columns:
                feature_dict[each] |= set(self.user_info[each].unique())
            for each in self.content_info.columns:
                feature_dict[each] |= set(self.content_info[each].unique())

            for key in feature_dict.keys():
                le = LabelEncoder()
                le.fit(list(feature_dict[key]))
                le_dict[key] = le
                feature_dict[key] = len(feature_dict[key])
            save_file=open("feature_dict.bin","wb")
            pickle.dump(feature_dict,save_file)
            pickle.dump(le_dict,save_file)
            save_file.close()
        self.feature_dict = feature_dict
        self.le_dict = le_dict

        del feature_dict
        del le_dict
        gc.collect()
        self.reader = pd.read_csv('../train_20190518.csv',header=None, iterator=True)

    def get_feature_dict():
        return self.feature_dict

    def get_feature_le():
        return self.le_dict

    def get_next():
        batch = self.reader.get_chunk(self.chunksize)
        batch.columns = ['label','uId','adId','operTime','siteId','slotId','contentId','netType']
        batch = batch.merge(user_info,on='uId',how='left')
        batch = batch.merge(ad_info,on='adId',how='left')
        batch = batch.merge(content_info,on='contentId',how='left')
        del batch['operTime']
        batch.fillna(-1,inplace=True)
        for key in self.le_dict.keys():
            batch[key] = self.le_dict.transform(batch[key])
        return batch

    def get_test():
        test = pd.read_csv('../test_20190518.csv',header=None)
        test.columns = ['label','uId','adId','operTime','siteId','slotId','contentId','netType']
        del test['operTime']
        test = test.merge(user_info,on='uId',how='left')
        test = test.merge(ad_info,on='adId',how='left')
        test = test.merge(content_info,on='contentId',how='left')
        test.fillna(-1,inplace=True)
        return test


    
