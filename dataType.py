import pandas as pd
import os
from tqdm import tqdm
import time
import datatable as dt

def replace_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')]
    for i in tqdm(range(len(files))):
        file = files[i]
        data = pd.read_hdf(file)
        os.remove(file)
        data = data.astype('float32')
        file = file.replace('.h5', '.jay')
        dt.Frame(data).to_jay(file)
        # data.to_feather(file)

def dataType_t(folder):
    start_time = time.time()
    dataType = '.jay'
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(dataType)]
    for i in tqdm(range(len(files))):
        file = files[i]
        # data = pd.read_hdf(file)
        # print(data)
        # time.sleep(100)
        data = dt.fread(file)
        os.remove(file)
        data = data.to_pandas()
        data = data.astype('float32')
        dt.Frame(data).to_jay(file)
        # data.to_hdf(file, key='data', mode='w')
    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':
    dataType_t('C:/Users/86155/PycharmProjects/Stock/growth_death_data2/test')