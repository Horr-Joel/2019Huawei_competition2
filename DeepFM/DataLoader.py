import pandas as pd 
import numpy as np
import config
import gc
import os
import pickle
from sklearn.preprocessing import LabelEncoder 
from collections import defaultdict as dd
# 3-7 7-13  13-16  16-20 21-3
def operTime_map(x):
    if x>=3 and x<7:
        return 0
    elif x >=7 and x<13:
        return 1
    elif x>=13 and x<16:
        return 2
    elif x>=16 and x<20:
        return 3
    else:
        return 4

class DataLoader():
    def __init__(self,chunksize, **kwargs):
        self.chunksize = chunksize
        # has feature nunique dict and feature labelencoder dict？
        has_dict = False
        
        print('load feature dict...')
        if os.path.exists('../data/feature_dict.bin'):
            has_dict = True
            load_file = open("../data/feature_dict.bin","rb")
            feature_dict = pickle.load(load_file)
            le_dict = pickle.load(load_file)
            load_file.close()
        else:
            feature_dict = dd(set)
            le_dict = {}
            reader = pd.read_csv(config.TRAIN_FILE,header=None, chunksize=self.chunksize)
            
            
            for each in ['uId','adId','operTime','siteId','slotId','contentId','netType']:
                feature_dict[each] = set()
            for each in reader:
                each.columns = ['label','uId','adId','operTime','siteId','slotId','contentId','netType']
                each.fillna(-1,inplace=True)

                each['operTime'] = each['operTime'].apply(lambda x: operTime_map(int(x.split(' ')[1][:2])) if x!=-1 else x)

                for f in ['uId','adId','operTime','siteId','slotId','contentId','netType']:
                    feature_dict[f] |= set(each[f].unique())
            
        print('load ad info...')
        self.ad_info = pd.read_csv('../data/ad_info.csv',header=None)
        self.ad_info.columns = ['adId','billId','primId','creativeType','intertype','spreadAppId']
        self.ad_info.fillna(-1,inplace=True)
        for f in ['adId','primId','creativeType','intertype','spreadAppId']:
            self.ad_info[f] = self.ad_info[f].astype(np.int32)

        print('load user info...')
        self.user_info = pd.read_csv('../data/user_info.csv',header=None)
        self.user_info.columns = ['uId','age','gender','city','province','phoneType','carrier']
        for f in ['age','gender','city','province','phoneType','carrier']:
            self.user_info[f] = self.user_info[f].astype(np.float32)
        self.user_info.fillna(-1,inplace=True)

        print('load content info...')
        self.content_info = pd.read_csv('../data/content_info.csv',header=None)
        self.content_info.columns=['contentId','firstClass','secondClass']
        self.content_info['contentId'] = self.content_info['contentId'].astype(np.int32)
        #content_info['secondClass'] = content_info['secondClass'].apply(lambda x: x.split('#') if isinstance(x,str) else [])
        self.content_info.fillna('Nan',inplace=True)

        self.content_info['firstClass'] = LabelEncoder().fit_transform(self.content_info['firstClass'])
        self.content_info['secondClass'] = LabelEncoder().fit_transform(self.content_info['secondClass'])

        if not has_dict:
            for each in self.ad_info.columns:
                feature_dict[each] |= set(self.ad_info[each].unique())
            for each in self.user_info.columns:
                feature_dict[each] |= set(self.user_info[each].unique())
            for each in self.content_info.columns:
                feature_dict[each] |= set(self.content_info[each].unique())|set([-1])

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
        self.reader = pd.read_csv(config.TRAIN_FILE,header=None, iterator=True)


    def get_feature_dict(self):
        return self.feature_dict

    def get_feature_le(self):
        return self.le_dict

    def get_next(self):
        batch = self.reader.get_chunk(self.chunksize)
        batch.columns = ['label','uId','adId','operTime','siteId','slotId','contentId','netType']
        batch = batch.merge(self.user_info,on='uId',how='left')
        batch = batch.merge(self.ad_info,on='adId',how='left')
        batch = batch.merge(self.content_info,on='contentId',how='left')
        
        batch['operTime'] = batch['operTime'].apply(lambda x: operTime_map(int(x.split(' ')[1][:2])))

        batch.fillna(-1,inplace=True)
        for key in self.le_dict.keys():
            batch[key] = self.le_dict[key].transform(batch[key])
        return batch

    def get_test(self):
        test = pd.read_csv(config.TEST_FILE,header=None)
        test.columns = ['label','uId','adId','operTime','siteId','slotId','contentId','netType']

        test['operTime'] = test['operTime'].apply(lambda x: operTime_map(int(x.split(' ')[1][:2])))

        test = test.merge(user_info,on='uId',how='left')
        test = test.merge(ad_info,on='adId',how='left')
        test = test.merge(content_info,on='contentId',how='left')
        test.fillna(-1,inplace=True)
        return test