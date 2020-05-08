# MFCC

import os
import numpy as np
import scipy
from scipy.io import wavfile
# import scipy.fftpack as fft
from scipy.signal import get_window
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.fftpack import dct as sdct
import math
import pickle

def maxx(x):
    maxx=0
    for i in range(len(x)):
        if x[i]>maxx:
            maxx=x[i]
    return maxx

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

def normalize_audio(audio):
    ab=np.abs(audio)
    audio = audio / maxx(ab)
    return audio

def pad(A,l):
#     A=A.reshape(len(A),1)
    for j in range(l):
#         A=np.insert(A,A.shape[1],2,axis=1)
        A=np.insert(A,A.shape[0],2,axis=0)
#         A=np.insert(A,0,2,axis=1)
        A=np.insert(A,0,2,axis=0)
#     print("padding done!")
    return A

def frame_audio(audio, FFT_size=2048,sample_rate=44100):
    # hop_size in ms
    hop_size=15
    audio = pad(audio,FFT_size//2)
    frame_len = (sample_rate*hop_size)//1000
    frame_num = (len(audio) - FFT_size)//frame_len
    frames = np.ones((frame_num+1,FFT_size))
    
    for n in range(frame_num+1):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames

def compute_fft(audio_win,FFT_size):
    audio_winT = audio_win.T
    col=audio_winT.shape[1]
    row=1 + FFT_size // 2
    audio_fft = np.zeros((row, col), dtype=np.complex64)

    for n in range(col):
        f=fft(audio_winT[:, n]) #in-built fft
        app = f[:row]
        audio_fft[:, n] = app

    audio_fft = audio_fft.T
    return audio_fft

def freq_to_mel(freq):
    a=freq/700
    b=a+1
    x=math.log10(b)
    y=2595*x
    return y

def met_to_freq(mels):
    a=mels/2595
    b=pow(10,a)
    c=b-1
    d=700*c
    return d

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
#     print("MEL min: {0}".format(fmin_mel))
#     print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
#     print(mels,mels2)
    freqs = met_to_freq(mels)
    a = FFT_size + 1
    b = a/sample_rate
    c = b*freqs
    d = []
    for i in range(len(c)):
        d.append(math.floor(c[i]))
    d = np.asarray(d)
    return d, freqs

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = linspace(1,0, filter_points[n + 1] - filter_points[n]-1)
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = linspace(1,0, filter_points[n + 2] - filter_points[n+1]-1,True)
        
    return filters



mfcc_X = []
mfcc_Y = []

for k in range(10):
    curr=nums[k]
    s = "Dataset/training/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
        ind = ind+1
        aud = s + f
        sample_rate, samples = wavfile.read(aud)
        audio=samples
        audio = normalize_audio(audio)

        hop_size = 15 #ms
        FFT_size = 2048

        audio_framed = frame_audio(audio, FFT_size=FFT_size, sample_rate=sample_rate)

        window = get_window("hann", FFT_size)

        audio_win = audio_framed * window

        audio_fft = compute_fft(audio_win,FFT_size)

        audio_power = np.square(np.abs(audio_fft))

        freq_min = 0
        freq_high = sample_rate / 2 #nyquest limit
        mel_filter_num = 10

        filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

        filters = get_filters(filter_points, FFT_size)

        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]

        audio_filtered = np.dot(filters, audio_power.T)
        d = [[0 for i in range(audio_filtered.shape[1])] for j in range(audio_filtered.shape[0])]
        for i in range(audio_filtered.shape[0]):
            for j in range(audio_filtered.shape[1]):
                try:
                    d[i][j]=10.0*math.log10(audio_filtered[i][j])
                except:
                    d[i][j]=0
        d = np.asarray(d)
        audio_log=d

        dct_filter = sdct(audio_log)

        cepstral_coefficents = np.dot(dct_filter.T, audio_log)
        
        mfcc_X.append(cepstral_coefficents)
        mfcc_Y.append(k)
        
        if ind==10:
            break

for k in range(10):
    curr=nums[k]
    s = "Dataset/training/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
        ind = ind+1
        aud = s + f
        sample_rate, samples = wavfile.read(aud)
        rn = rnd.randint(0,6)
        noi = noise[rn]
        sn, noi = wavfile.read("Dataset/_background_noise_/"+noi)
        rn_n = rnd.choice(noi,len(samples))
        audio = 0.95*samples + 0.5*rn_n

        audio = normalize_audio(audio)

        hop_size = 15 #ms
        FFT_size = 2048

        audio_framed = frame_audio(audio, FFT_size=FFT_size, sample_rate=sample_rate)

        window = get_window("hann", FFT_size)

        audio_win = audio_framed * window

        audio_fft = compute_fft(audio_win,FFT_size)

        audio_power = np.square(np.abs(audio_fft))

        freq_min = 0
        freq_high = sample_rate / 2 #nyquest limit
        mel_filter_num = 10

        filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

        filters = get_filters(filter_points, FFT_size)

        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]

        audio_filtered = np.dot(filters, audio_power.T)
        d = [[0 for i in range(audio_filtered.shape[1])] for j in range(audio_filtered.shape[0])]
        for i in range(audio_filtered.shape[0]):
            for j in range(audio_filtered.shape[1]):
                try:
                    d[i][j]=10.0*math.log10(audio_filtered[i][j])
                except:
                    d[i][j]=0
        d = np.asarray(d)
        audio_log=d

        dct_filter = sdct(audio_log)

        cepstral_coefficents = np.dot(dct_filter.T, audio_log)
        
        mfcc_X.append(cepstral_coefficents)
        mfcc_Y.append(k)
        
        if ind==10:
            break

vmfcc_X = []
vmfcc_Y = []

for k in range(10):
    curr=nums[k]
#     print(k)
    s = "Dataset/validation/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
        ind = ind+1
        aud = s + f
        sample_rate, samples = wavfile.read(aud)
        audio=samples
        audio = normalize_audio(audio)

        hop_size = 15 #ms
        FFT_size = 2048

        audio_framed = frame_audio(audio, FFT_size=FFT_size, sample_rate=sample_rate)

        window = get_window("hann", FFT_size)

        audio_win = audio_framed * window

        audio_fft = compute_fft(audio_win,FFT_size)

        audio_power = np.square(np.abs(audio_fft))

        freq_min = 0
        freq_high = sample_rate / 2 #nyquest limit
        mel_filter_num = 10

        filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

        filters = get_filters(filter_points, FFT_size)

        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]

        audio_filtered = np.dot(filters, audio_power.T)
        d = [[0 for i in range(audio_filtered.shape[1])] for j in range(audio_filtered.shape[0])]
        for i in range(audio_filtered.shape[0]):
            for j in range(audio_filtered.shape[1]):
                try:
                    d[i][j]=10.0*math.log10(audio_filtered[i][j])
                except:
                    d[i][j]=0
        d = np.asarray(d)
        audio_log=d

        dct_filter = sdct(audio_log)

        cepstral_coefficents = np.dot(dct_filter.T, audio_log)
        
        vmfcc_X.append(cepstral_coefficents)
#         print(k)
        vmfcc_Y.append(k)
        
        if ind==5:
            break

for k in range(10):
    curr=nums[k]
#     print(k)
    s = "Dataset/validation/"+curr+"/"
    ind = 0
    for f in os.listdir(s):
        ind = ind+1
        aud = s + f
        sample_rate, samples = wavfile.read(aud)
        rn = rnd.randint(0,6)
        noi = noise[rn]
        sn, noi = wavfile.read("Dataset/_background_noise_/"+noi)
        rn_n = rnd.choice(noi,len(samples))
        samples = 0.95*samples + 0.5*rn_n
        audio=samples
        audio = normalize_audio(audio)

        hop_size = 15 #ms
        FFT_size = 2048

        audio_framed = frame_audio(audio, FFT_size=FFT_size, sample_rate=sample_rate)

        window = get_window("hann", FFT_size)

        audio_win = audio_framed * window

        audio_fft = compute_fft(audio_win,FFT_size)

        audio_power = np.square(np.abs(audio_fft))

        freq_min = 0
        freq_high = sample_rate / 2 #nyquest limit
        mel_filter_num = 10

        filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

        filters = get_filters(filter_points, FFT_size)

        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]

        audio_filtered = np.dot(filters, audio_power.T)
        d = [[0 for i in range(audio_filtered.shape[1])] for j in range(audio_filtered.shape[0])]
        for i in range(audio_filtered.shape[0]):
            for j in range(audio_filtered.shape[1]):
                try:
                    d[i][j]=10.0*math.log10(audio_filtered[i][j])
                except:
                    d[i][j]=0
        d = np.asarray(d)
        audio_log=d

        dct_filter = sdct(audio_log)

        cepstral_coefficents = np.dot(dct_filter.T, audio_log)
        
        vmfcc_X.append(cepstral_coefficents)
#         print(k)
        vmfcc_Y.append(k)
        
        if ind==5:
            break


pickle_out = open("mfcc.pickle","wb")
pickle.dump(mfcc_X, pickle_out)
pickle_out.close()

pickle_out = open("vmfcc.pickle","wb")
pickle.dump(vmfcc_X, pickle_out)
pickle_out.close()