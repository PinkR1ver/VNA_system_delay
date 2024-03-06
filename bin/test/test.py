import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt 
from rich.progress import track
from scipy import io
import matplotlib.pyplot as plt

def logmag2liner(x):
    return 10 ** (x/20)

start_index = 3161
t = np.linspace(0, 4 * 1e-8, int(1e5))


if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)
    
    data_path = os.path.join(base_path, '4.S2P')
    mc_path = os.path.join(base_path, 'MC.S2P')
    t0 = pd.read_csv(os.path.join(base_path, 't0.csv'))
    t0 = t0['t0'].values[0]
    
    df = pd.read_csv(mc_path, skiprows=5, delimiter='	', names=['Frequency', 'S11_amp', 'S11_phase', 'S21_amp', 'S21_phase', 'S12_amp', 'S12_phase', 'S22_amp', 'S22_phase'])
    
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
    
    df = pd.read_csv(data_path, skiprows=5, delimiter='	', names=['Frequency', 'S11_amp', 'S11_phase', 'S21_amp', 'S21_phase', 'S12_amp', 'S12_phase', 'S22_amp', 'S22_phase'])
    
    
    frequency = df['Frequency'].values
    phase = df['S21_phase'].values
    amp = df['S21_amp'].values
    
    frequency = frequency[start_index:-1]
    phase = phase[start_index:-1]
    amp = amp[start_index:-1]
    
    signal = np.zeros(len(t))
    count = 0
    
    
    for ph, am, freq in zip(phase, amp, frequency):
        signal += logmag2liner(am) * np.cos(2 * np.pi * freq * t + ph/180 * np.pi)
        count += 1
        
    signal = signal / count
    signal = signal - mc_signal
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal, label='Signal')
    plt.xlabel('Time(s)')
    plt.ylabel('Magnitude')
    plt.show()
    
    t2_index = np.argmax(signal)
    t2 = t[t2_index]
    
    t1 = t2 - t0
    distance = 0.5 * 3e8 * t1 * 1e2
    
    print(distance)
    
    
    
    