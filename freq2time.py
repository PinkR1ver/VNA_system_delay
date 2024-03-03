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
    fig_path = os.path.join(base_path, 'fig')
    signal_path = os.path.join(base_path, 'signal')
    
    start_index = 3161
    signal_df = pd.DataFrame()
    
    t = np.linspace(0, 4 * 1e-8, int(1e5))
    np.savetxt(os.path.join(signal_path, 't.txt'), t)
    labels = []
    
    signal_df['t'] = t
    
    
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
                np.savetxt(os.path.join(signal_path, 'MC.txt'), mc_signal)
                # make mc_signal as a column to signal_df
                signal_df['MC'] = mc_signal
                
                # plt.figure(figsize=(10, 5))
                # plt.plot(t, mc_signal, label='MC')
                # plt.xlabel('Time(s)')
                # plt.ylabel('Magnitude')
                # plt.title('MC Signal')
                # plt.savefig(os.path.join(fig_path, 'MC_Signal.png'))
                # plt.show()
                
                print('MC Signal Compute Done')

    for root, dirs, files in os.walk(data_path):
        for file in track(files, total=len(files)):
            
            if "S2P" in file and 'MC' not in file:
                
                label = file.split('.')[0]
                label = int(label)
                labels.append(label)

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

                np.savetxt(os.path.join(signal_path, f'{label}.txt'), signal)
                signal_df[label] = signal

                # plt.plot(t, signal, label=label)
                # plt.plot(t, mc_signal, label='MC')
                # plt.xlabel('Time(s)')
                # plt.ylabel('Magnitude')
                # plt.title('Signal')
                # plt.savefig(os.path.join(fig_path, f'Signal_{label}cm.png'))
                # plt.show()
                
    signal_df.to_excel(os.path.join(signal_path, 'signal.xlsx'), index=False)