import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from rich.progress import track


if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'signal', 'time.csv')
    save_path = os.path.join(base_path, 'offset')
    
    dataset = pd.read_csv(data_path)
    t0_precision = 100
    
    
    for i in track(range(1, 9), description='Calculating...'):
        
        # get the combination C(n,i)
        
        combs = itertools.combinations(dataset.distance, i)
        offset_global = pd.DataFrame()
        
        for comb in combs:
            
            t0_list = []
            t2_list = []
            distance_list = []
            
            for j in comb:
                
                row = dataset.loc[dataset['distance'] == j]
                t0_list.append(row['t0'].values[0])
                t2_list.append(row['t2'].values[0])
                distance_list.append(row['distance'].values[0])
            
            t0_range = np.linspace(min(t0_list) * 0.95, max(t0_list) * 1.05, t0_precision)
            offset_list = []
            
            for t0 in t0_range:
                
                offset_tmp = []
                
                for j, t2 in enumerate(t2_list):
                    
                    truth = distance_list[j]
                    
                    t1 = t2 - t0
                    predict = t1 * 3e8 / 2 * 1e2
                    
                    offset = abs(predict - truth) / truth
                    offset_tmp.append(offset)
                
                offset_list.append(np.mean(offset_tmp))
            
            best_index = np.argmin(offset_list)
            t0 = t0_range[best_index]
            
            offset_list = []
            
            for j in range(len(dataset)):
                
                truth = dataset['distance'].values[j]
                t2 = dataset['t2'].values[j]
                t1 = t2 - t0
                predict = t1 * 3e8 / 2 * 1e2
                
                offset = abs(predict - truth) / truth
                offset_list.append(offset)
            
            offset = np.mean(offset_list)    
            
            
            comb = list(comb)
            comb_str = str(comb[0])
            for j in range(1, len(comb)):
                comb_str += (f'_{comb[j]}')
            comb_str = [comb_str]
            offset_global = pd.concat([offset_global, pd.DataFrame({'sample': comb_str, 't0': t0, 'offset': offset})], ignore_index=True)
        
        offset_global.to_csv(os.path.join(save_path, f'offset_{i}.csv'), index=False)
            
                        
                
                
                
                
                
                