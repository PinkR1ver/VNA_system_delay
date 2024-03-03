import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'signal')
    
    peaks_list = []
    
    signal_df = pd.read_excel(os.path.join(data_path, 'signal.xlsx'))
    
    t = signal_df['t'].values
    mc_signal = signal_df['MC'].values
    t0_list = []
    
    figure = plt.figure()
    
    train_df = pd.DataFrame()
    
    for i in range(2, len(signal_df.columns)):
        
        distance = int(signal_df.columns[i])
        
        signal = signal_df.iloc[:, i].values
        signal = signal - mc_signal
        
        peak = np.argmax(signal)
        plt.plot(t, signal)
        plt.plot(t[peak], signal[peak], "x")
        
        t2 = t[peak]
        t1 = distance * 1e-2 * 2 / 3e8
        t0 = t2 - t1
        t0_list.append(t0)
        
        train_df[distance] = [t0, t1, t2]
        
        
    plt.show()
    
    # save to csv want index be 't0', 't1', 't2', and add distance as column name
    train_df = train_df.T
    train_df.columns = ['t0', 't1', 't2']
    train_df.index.name = 'distance'
    train_df.to_csv(os.path.join(data_path, 'time.csv'))
    
    
    
    t0 = np.mean(t0_list)
    print(f't0: {t0:.2e}s')
    
    print()
    
    offset_list = []
    
    for i in range(2, len(signal_df.columns)):
        
        distance = int(signal_df.columns[i])
        
        signal = signal_df.iloc[:, i].values
        signal = signal - mc_signal
        
        peak = np.argmax(signal)
        
        prediction = (t[peak] - t0) * 3e8 / 2 * 100
        offset = abs(distance - prediction) / distance
        offset_list.append(offset)
        
        print(f'{distance} cm: {prediction} cm')
        
    print()
    print(f'Average offset: {np.mean(offset_list) * 100:.2f}%')
        
        
        
    
        
        
        
        