import os
from tqdm import tqdm
import pandas as pd

def clean_financial_forecast_train_data(folder):
    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    for i in tqdm(range(len(csv_files))):
        file = csv_files[i]
        data = pd.read_csv(file)
        # 不用-256，而用-257多一天就是为了删除第一个重复的
        fore_data = data.iloc[-257:]
        fact_data = fore_data.iloc[:, 1:3].join(fore_data.iloc[:, 15:-12])
        fact_data = fact_data.drop_duplicates(subset=['numberOfShares', 'dividend','revenue','costOfRevenue',
                                                      'capitalExpenditure','freeCashFlow'], keep='first')
        fact_data = fact_data.reset_index()
        fact_data.drop(columns=['index'], inplace=True)
        if fact_data.shape[0] < 5:
            print(file)
            os.remove(file)

if __name__ == '__main__':
    clean_financial_forecast_train_data('financial_daily_data/test')