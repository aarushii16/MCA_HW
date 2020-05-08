# Spectogram

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import math
import os

def get_xn(ts):
    mag = []
    L2 = len(ts)//2
    L  = len(ts)
    for n in range(L2):
        ks = np.asarray([i for i in range(L)])
        num = complex(0,2)
        a=num*ks*math.pi
        b=a*n
        c=b/L
        d=np.exp(c)
        e=np.sum(ts*d)/L
        ll = abs(e)*2
        mag.append(ll)
    return(mag)

def linspace(end,start=0,step=9,reverse=False):
    x = end/step
    re=[]
    if reverse==False:
        for i in range(start,step+1):
            re.append(i*x)
        return np.asarray(re)
    else:
        for i in range(step,start-1,-1):
            re.append(i*x)
        return np.asarray(re)

def get_Hz_scale_vec(ks,sample_rate,Npoints):
    freq_Hz = np.asarray(ks*sample_rate/Npoints)
    freq_Hz  = freq_Hz.astype('int')
    return(freq_Hz )

def create_spectrogram(ts,win_s):
    noverlap = win_s//2
    starts=[]
    var=0
    while var*noverlap<len(ts):
        starts.append(int(var*noverlap))
        var=var+1
    starts=np.asarray(starts)
    final=[]
    for i in range(len(starts)):
        if starts[i]+win_s<len(ts):
            final.append(starts[i])
    starts=final
    xns = []
    for start in starts:
        win=ts[start:start + win_s]
        ts_window = get_xn(win) 
        xns.append(ts_window)
    specX = np.array(xns).T
    
    m=[[0 for i in range(specX.shape[1])] for j in range(specX.shape[0])]
    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            try:
                m[i][j]=math.log10(specX[i][j])
            except:
                m[i][j]=0
    return(starts,np.asarray(m))

def plot_spectrogram(spec,ks,sample_rate, L, starts):
    plt.figure(figsize=(20,8))
    plt_spec = plt.imshow(spec,origin='lower')

    ks      = linspace(spec.shape[0])
    ksHz    = get_Hz_scale_vec(ks,sample_rate,len(ts))
    plt.yticks(ks,ksHz)
    plt.ylabel("Frequency (Hz)")

    ts_spec = linspace(spec.shape[1])
    mat=linspace(total_ts_sec*starts[-1]/len(ts))
    mat=[round(mat[i],2) for i in range(len(mat))]
    plt.xticks(ts_spec,mat)
    plt.xlabel("Time (sec)")
    st="Spectrogram window size= "+str(win_s)+" Spectogram Shape= "+str(spec.shape)
    plt.title(st)
    plt.show()
    return(plt_spec)




spec_X = []
spec_Y = []

vspec_X = []
vspec_Y = []

nums = ["zero","one","two","three","four","five","six","seven","eight","nine"]

noise = ["doing_the_dishes.wav","dude_miaowing.wav","exercise_bike.wav","pink_noise.wav","running_tap.wav","white_noise.wav"]

import numpy.random as rnd
from IPython.display import Audio

for i in range(10):
    curr=nums[i]
    s = "Dataset/training/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
#         print(f)
        ind = ind+1
        aud = s + f
        sample_rate,samples = wavfile.read(aud)
        
        ts=np.asarray(samples)
        total_ts_sec = len(ts)/sample_rate

        mag = get_xn(ts)

        ks   = linspace(len(mag))
        ksHz = get_Hz_scale_vec(ks,sample_rate,len(ts))

        win_s = 256
        starts, spec = create_spectrogram(ts,win_s)
        
        spec_X.append(spec)
        spec_Y.append(i)
        
        if ind==100:
            break

for i in range(10):
    curr=nums[i]
    s = "Dataset/training/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
#         print(f)
        ind = ind+1
        aud = s + f
        sample_rate,samples = wavfile.read(aud)
        rn = rnd.randint(0,6)
        noi = noise[rn]
        sn, noi = wavfile.read("Dataset/_background_noise_/"+noi)
        rn_n = rnd.choice(noi,len(samples))
        samples = 0.95*samples + 0.5*rn_n
        
        ts=np.asarray(samples)
        total_ts_sec = len(ts)/sample_rate

        mag = get_xn(ts)

        ks   = linspace(len(mag))
        ksHz = get_Hz_scale_vec(ks,sample_rate,len(ts))

        win_s = 256
        starts, spec = create_spectrogram(ts,win_s)
        
        spec_X.append(spec)
        spec_Y.append(i)
        
        if ind==10:
            break

#         plot_spectrogram(spec,ks,sample_rate,win_s,starts)

Audio(samplesn, rate=sample_rate)

for i in range(10):
    curr=nums[i]
    s = "Dataset/validation/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
#         print(f)
        ind = ind+1
        aud = s + f
        sample_rate,samples = wavfile.read(aud)


        ts=np.asarray(samples)
        total_ts_sec = len(ts)/sample_rate

        mag = get_xn(ts)

        ks   = linspace(len(mag))
        ksHz = get_Hz_scale_vec(ks,sample_rate,len(ts))

        win_s = 256
        starts, spec = create_spectrogram(ts,win_s)
        
        vspec_X.append(spec)
        vspec_Y.append(i)
        
        if ind==10:
            break

for i in range(10):
    curr=nums[i]
    s = "Dataset/validation/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
#         print(f)
        ind = ind+1
        aud = s + f
        sample_rate,samples = wavfile.read(aud)
        rn = rnd.randint(0,6)
        noi = noise[rn]
        sn, noi = wavfile.read("Dataset/_background_noise_/"+noi)
        rn_n = rnd.choice(noi,len(samples))
        samples = 0.95*samples + 0.5*rn_n

        ts=np.asarray(samples)
        total_ts_sec = len(ts)/sample_rate

        mag = get_xn(ts)

        ks   = linspace(len(mag))
        ksHz = get_Hz_scale_vec(ks,sample_rate,len(ts))

        win_s = 256
        starts, spec = create_spectrogram(ts,win_s)
        
        vspec_X.append(spec)
        vspec_Y.append(i)
        
        if ind==5:
            break

#         plot_spectrogram(spec,ks,sample_rate,win_s,starts)