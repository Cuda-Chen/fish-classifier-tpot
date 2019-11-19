from tpot import TPOTClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

num_classes = 42 # 41 + 1
input_size = 122 # 10 + 16 * 3 + 64
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
           41, 42] 
class_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42']
lbp_train = pd.read_csv('./LBP_feature_train.csv')
lbp_val = pd.read_csv('./LBP_feature_val.csv')
fd_train = pd.read_csv('./fd_feature_train.csv')
fd_val = pd.read_csv('./fd_feature_val.csv')
color_som_train = pd.read_csv('./color_som_train.csv')
color_som_val = pd.read_csv('./color_som_val.csv')
train_label = pd.read_csv('./train_label.csv')
val_label = pd.read_csv('./val_label.csv')

# get your training and testing data here
# and to make sure to reshape and normalize!
x_train = pd.concat([lbp_train, fd_train, color_som_train], axis=1).values
x_test = pd.concat([lbp_val, fd_val, color_som_val], axis=1).values

# convert class vectors to labels
# there are some missing class!
y_train = train_label['class_no'].values
y_test = val_label['class_no'].values

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

tpot =  TPOTClassifier(verbosity=2, n_jobs=-1)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
