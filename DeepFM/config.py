

#set the path to files
TRAIN_FILE = '../data/train_20190518.csv'
TEST_FILE = '../data/test_20190518.csv'

MODEL_FILE = './model/model.h5'
SUB_DIR = './output'

RANDOMSTATE = 2019

EPOCH = 1
BATCH_SIZE = 256
EMBEDDING_SIZE = 8

TRAIN_BATCH_SIZE = 10000000

CATEGORECIAL_COLS =[
    'uId','adId','siteId','slotId','contentId','netType',
    'firstClass','secondClass','age','gender','city','province','phoneType','carrier',
    'billId','primId','creativeType','intertype','spreadAppId','operTime'
]

NUMERIC_COLS = [

]

IGNORE_COLS = [
    
]
