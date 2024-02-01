import numpy as np

# 加载 .npy 文件到内存中
# train_data = np.load(
#     'F:\\研究\\paper1\\code\\feature_extraction\\data\\train_data_densenet121_TEST.npy')
train_data = np.load('F:\\研究\\paper1\\code\\feature_extraction\\data_rubbish\\data\\train_data3_densenet121.npy')

# val_data = np.load('F:\\研究\\paper1\\code\\feature_extraction\\data\\val_data_densenet121_TEST.npy')
val_data = np.load('F:\\研究\\paper1\\code\\feature_extraction\\data_rubbish\\data\\val_data3_densenet121.npy')
# 打印加载的数组
print(train_data.shape)
print("----")
print(train_data[1, :])


import scipy.io as sio
import numpy as np

# 加载 .npy 文件到内存中
# data = np.load('filename.npy')

# 定义要保存的 .mat 文件名和路径
save_name = 'F:\\研究\\paper1\\code\\feature_extraction\\data_rubbish\\data\\data_rubbish2_densenet121.mat'

# 将数据保存到 .mat 文件中
sio.savemat(save_name, {'train_data_densenet121': train_data, 'val_data': val_data})

