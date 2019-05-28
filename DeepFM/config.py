

#set the path to files
TRAIN_FILE = './data/train.csv'
TEST_FILE = './data/test.csv'

MODEL_FILE = './model/model.h5'
SUB_DIR = './output'

RANDOMSTATE = 2019


CATEGORECIAL_COLS =[
    'uId','adId','siteId','slotId','contentId','netType',
    'firstClass','secondClass','age','gender','city','province','phoneType','carrier',
    'billId','primId','creativeType','intertype'
]

NUMERIC_COLS = [

]

IGNORE_COLS = [
    'spreadAppId',
]
