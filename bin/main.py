import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt 
from rich.progress import track
from scipy import io

def logmag2liner(x):
    return 10 ** (x/20)


if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'data')
    
    start_index = 3161
    signal_df = pd.DataFrame()
    
    t = np.linspace(0, 4 * 1e-8, int(1e5))
    distance_list = []
    t0_list = []
    t2_list = []
    
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            
            if "MC.S2P" in file:

                df = pd.read_csv(os.path.join(root, file), skiprows=5, delimiter='	', names=['Frequency', 'S11_amp', 'S11_phase', 'S21_amp', 'S21_phase', 'S12_amp', 'S12_phase', 'S22_amp', 'S22_phase'])
                
                frequency = df['Frequency'].values
                phase = df['S21_phase'].values
                amp = df['S21_amp'].values
                
                frequency = frequency[start_index:-1]
                phase = phase[start_index:-1]
                amp = amp[start_index:-1]
                
                mc_signal = np.zeros(len(t))
                count = 0
                
                
                for ph, am, freq in zip(phase, amp, frequency):
                    mc_signal += logmag2liner(am) * np.cos(2 * np.pi * freq * t + ph/180 * np.pi)
                    count += 1
                    
                mc_signal = mc_signal / count
                
                print('MC Signal Compute Done!')

    for root, dirs, files in os.walk(data_path):
        for file in track(files, total=len(files)):
            
            if "S2P" in file and 'MC' not in file:
                
                distance = file.split('.')[0]
                distance = int(distance)
                distance_list.append(distance)

                df = pd.read_csv(os.path.join(root, file), skiprows=5, delimiter='	', names=['Frequency', 'S11_amp', 'S11_phase', 'S21_amp', 'S21_phase', 'S12_amp', 'S12_phase', 'S22_amp', 'S22_phase'])

                frequency = df['Frequency'].values
                phase = df['S21_phase'].values
                amp = df['S21_amp'].values
                
                frequency = frequency[start_index:-1]
                phase = phase[start_index:-1]
                amp = amp[start_index:-1]
                
                # plt.plot(frequency, logmag2liner(amp))
                # plt.show()
                
                signal = np.zeros(len(t))
                count = 0
                
                
                for ph, am, freq in zip(phase, amp, frequency):
                    signal += logmag2liner(am) * np.cos(2 * np.pi * freq * t + ph/180 * np.pi)
                    count += 1
                    
                signal = signal / count
                signal = signal - mc_signal
                
                t2_index = np.argmax(signal)
                t2 = t[t2_index]
                t1 = distance * 2 * 1e-2 / 3e8
                t0 = t2 - t1
                
                t2_list.append(t2)
                t0_list.append(t0)
                
    
    t0_range = np.linspace(min(t0_list) * 0.95, max(t0_list) * 1.05, 10000)
    
    offset_list = []
    
    for t0 in t0_range:
        
        offset_tmp = []
        
        for i, t2 in enumerate(t2_list):
            
            truth = distance_list[i]
            pred = 3e8 * (t2 - t0) / 2 * 1e2
            
            offset = abs(truth - pred) / truth
            
            offset_tmp.append(offset)
        
        offset_list.append(np.mean(offset_tmp))
    
    t0 = t0_range[np.argmin(offset_list)]
    
    print(f't0: {t0}')
    print(f'Best offset: {min(offset_list)}')
    
    t0 = pd.DataFrame({'t0': [t0]})
    t0.to_csv(os.path.join(base_path, 't0.csv'), index=False)