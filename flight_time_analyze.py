import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn


if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'offset')
    time_path = os.path.join(base_path, 'signal', 'time.csv')
    
    offset_list = []
    label_list = []
    offset_df = pd.DataFrame()
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if 'offset' in file and '1' not in file:
                
                dataset = pd.read_csv(os.path.join(root, file))
                label = file.split('.')[0].split('_')[-1]
                
                
                # offset_middle = np.percentile(dataset['offset'].values, 50)
                # offset_mean = np.mean(dataset['offset'].values)
                # offset_q1 = np.percentile(dataset['offset'].values, 25)
                # offset_q3 = np.percentile(dataset['offset'].values, 75)
                # offset_min = np.min(dataset['offset'].values)
                # offset_max = np.max(dataset['offset'].values)
                # offset = [offset_middle, offset_q1, offset_q3, offset_min, offset_max, offset_mean]
                
                # offset_list.append(offset)
                # label_list.append(label)
                offset = pd.DataFrame({'label': [label for i in range(len(dataset['offset'].values))] ,'offset': dataset['offset'].values, 'sample': dataset['sample'].values})
                offset_df = pd.concat([offset_df, offset], ignore_index=True)
     
    # show the boxplot and plot the mean vale as a line
    plt.figure()
    sns.boxplot(x="label", y="offset", data=offset_df)
    plt.plot(offset_df['label'].unique(), offset_df.groupby('label')['offset'].mean().values, 'r', label='Mean')
    plt.xlabel('Number of points choose')
    plt.ylabel('Offset')
    plt.legend()
    plt.show()
    
    best_index = np.argmin(offset_df['offset'].values)
    best_sample = offset_df['sample'].values[best_index]
    print(best_sample)
    
    plt.figure()
    sns.boxplot(data=offset_df['offset'].values, orient='h')
    plt.ylabel('Offset')
    plt.show()
    
    
    
    data_set = pd.read_csv(time_path)
    samples = best_sample.split('_')
    
    t0_list = []
    t2_list = []
    distance_list = []
    
    for sample in samples:
        t0 = data_set.loc[data_set['distance'] == int(sample)]['t0'].values[0]
        t2 = data_set.loc[data_set['distance'] == int(sample)]['t2'].values[0]
        distance = data_set.loc[data_set['distance'] == int(sample)]['distance'].values[0]
        
        t0_list.append(t0)
        t2_list.append(t2)
        distance_list.append(distance)
        
    t0_range = np.linspace(min(t0_list) * 0.95, max(t0_list) * 1.05, 10000)
    
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
        
    t0 = t0_range[np.argmin(offset_list)]
        
    # plot offset tmp and the minest point
    plt.figure()
    plt.plot(t0_range, offset_list)
    plt.plot(t0_range[np.argmin(offset_list)], min(offset_list), 'x')
    plt.xlabel('t0')
    plt.ylabel('Offset')
    plt.show()
    
    offset_list = []
    
    for i in range(len(data_set)):
            
            truth = data_set['distance'].values[i]
            t2 = data_set['t2'].values[i]
            t1 = t2 - t0
            predict = t1 * 3e8 / 2 * 1e2
            
            offset = abs(predict - truth) / truth
            offset_list.append(offset)
            
    offset = np.mean(offset_list)
    
    print(f't0: {t0_range[np.argmin(offset_list)]:.2e}s')
    print(f'Offset: {offset * 100:.2f}%')
    
    
    