from deepfm import KerasDeepFM
from DataLoader import DataLoader
import config
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import gc



if __name__ == "__main__":
    print("starting read data...")
    dl = DataLoader(10000000)
    LE_list = []


    if not LE_list:
        for f in batch.columns:
            LE_list.append(LabelEncoder())
            
    batch = dl.get_next()
    print("starting preprocessing data...")
    for col in config.CATEGORECIAL_COLS:
        lel = LabelEncoder()
        data[col] = lel.fit_transform(data[col])
    feat_dict = preprocess_data(data)
    
    train = data.iloc[:train_cols]
    test = data.iloc[train_cols:]
    del data
    gc.collect()
        
    
    x_train,x_val,y_train,y_val = train_test_split(train[config.NUMERIC_COLS+config.CATEGORECIAL_COLS],train['label'],test_size=0.8,random_state=config.RANDOMSTATE)
    x_train = x_train.values.T
    x_train = [np.array(x_train[i,:]) for i in range(x_train.shape[0])]
    y_train = y_train.values

    x_val = x_val.values.T
    x_val = [np.array(x_val[i,:]) for i in range(x_val.shape[0])]
    y_val = y_val.values


    print("train model...")
    kfm = KerasDeepFM(8, feat_dict)
    kfm.fit(x_train, y_train, x_val, y_val)

    if 1:
        sub = test['label']
        test = test[config.NUMERIC_COLS+config.CATEGORECIAL_COLS]
        test = test.values.T
        test = [np.array(test[i,:]) for i in range(test.shape[0])]
        predict = kfm.predict(test)

        sub['probability'] = predict
        sub.columns = ['id','probability']
        sub.to_csv('./output/submission.csv',index=False)



