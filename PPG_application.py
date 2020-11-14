import pandas as pd
import numpy as np
from joblib import load
import time
from biosppy.signals.tools import filter_signal
import sys
import re
import scipy.signal as signal
from scipy.signal import argrelmax
import heartpy as hp
from scipy import stats  
from heartpy.datautils import rolling_mean

def input_args():
    if 's' in sys.argv[1]:
        s = int(re.findall(r'\d+', sys.argv[1])[0]) #sample
    else:
        print("Input is not correct")
        exit()

    if 'f' in sys.argv[2]:
         f = int(re.findall(r'\d+', sys.argv[2])[0]) #filters: 1 is firwin, 2 is butterworth, 3 is bessel, and 4 is elliptic
    else:
        print("Input is not correct")
        exit()
		
    if 'p' in sys.argv[3]:
        p = int(re.findall(r'\d+', sys.argv[3])[0]) #peak detection: 1 
    else:
        print("Input is not correct")
        exit()
		
    if 'm' in sys.argv[4]:
        m = int(re.findall(r'\d+', sys.argv[4])[0]) #machine learning: 1 is SVM_L, 2 is SVM_RBF, 3 is ANN, 4 is KMeans
    else:
        print("Input is not correct")
        exit()
    return(s,f,p,m)

def _filtering(signal,fs,pass_frequency,n): 
    if n == 1:
        order = int(0.3 * fs)
        filtered, _, _ = filter_signal(signal=signal,ftype='FIR',band='bandpass',order=order,frequency=pass_frequency,sampling_rate=fs)
    elif n == 2:
        order = 5
        filtered, _, _ = filter_signal(signal=signal,ftype='butter',band='bandpass',order=order,frequency=pass_frequency,sampling_rate=fs)
    elif n == 3:
        order = 5
        filtered, _, _ = filter_signal(signal=signal,ftype='bessel',band='bandpass',order=order,frequency=pass_frequency,sampling_rate=fs)
    elif n == 4:
        order = 5
        filtered, _, _ = filter_signal(signal=signal,ftype='ellip',band='bandpass',order=order,frequency=pass_frequency,sampling_rate=fs,rp=5,rs=40)
    return(filtered)

		
def ML(features,clf):
	X = features.reshape(1,-1)
	y_pred = clf.predict(X)
	return(y_pred)

####################
    #Peak detection 
####################  
def _Peak_detection(filt_sig_sample, fs, n):
    if n == 1:
        NN_index_sig = np.array(signal.argrelextrema(filt_sig_sample, np.greater)).reshape(1,-1)[0]
        f, ppg_den = signal.periodogram(filt_sig_sample, fs)
        min_f = np.where(f >= 0.6)[0][0] 
        max_f = np.where(f >= 3.0)[0][0] 
        ppgHRfreq = ppg_den[min_f:max_f]
        HRfreq = f[min_f:max_f]    
        HRf = HRfreq[np.argmax(ppgHRfreq)]
        boundary = 0.5
        if HRf - boundary > 0.6:
            HRfmin = HRf - boundary
        else:
            HRfmin = 0.6
        if HRf + boundary < 3.0:
            HRfmax = HRf + boundary
        else:
            HRfmax = 3.0
        filtered = _ButterFilt(filt_sig_sample,fs,np.array([HRfmin,HRfmax]),5,'bandpass')
        NN_index_filtered = np.array(signal.argrelextrema(filtered, np.greater)).reshape(1,-1)[0]
        rpeak = np.array([]).astype(int)
        for i in NN_index_filtered:
            rpeak = np.append(rpeak,NN_index_sig[np.abs(i - NN_index_sig).argmin()])
        rpeak = np.unique(rpeak)
    elif n==2:
        rol_mean = rolling_mean(filt_sig_sample, windowsize = 0.75, sample_rate = fs)
        working_data = hp.peakdetection.fit_peaks(filt_sig_sample, rol_mean, fs)
        rpeak = working_data['peaklist']         
    elif n==3:
        rpeak = argrelmax(np.array(filt_sig_sample))[0]
    return(rpeak)
    
def _segmentation_heartCycle(filtsig,NN_index):
    MM_index = np.array([]).astype(int)
    for i in range(NN_index.shape[0]-1):
        MM_index = np.append(MM_index,np.argmin(filtsig[NN_index[i]:NN_index[i+1]]) + NN_index[i])
    return(MM_index)
    
def _ButterFilt(sig,fs,fc,order,btype):
    w = fc/(fs/2)
    b, a = signal.butter(order, w, btype =btype, analog=False)
    filtered = signal.filtfilt(b, a, sig)
    return(filtered)
    
def _range(x):
    r = np.round((np.amax(x)-np.amin(x)),3)
    return(r)
         
def __main__():
	####################
	   #Input args 
	####################
    (s,f,p,m) = input_args()
	####################
	   #Read file 
	####################
    t = time.time()
    filename_read = 'data_test/sample_' + str(s) + '.csv'
    sig_sample = pd.read_csv(filename_read)
    fs = 25
    filename_zscore_read = 'myModel1/zscore.csv'
	#
    df_z = pd.read_csv(filename_zscore_read)
	#
    if m == 1:
        modelName = 'SVM_L'
    elif m == 2:	
        modelName = 'SVM_RBF'
    elif m == 3:
        modelName = 'ANN'
    elif m == 4:
        modelName = 'KMeans'
    clf = load('myModel1/' + modelName + '_' + 'F' + str(f) + '_' + 'P' + str(p) + '.joblib')
	#
    loading_time = (time.time() - t)*1000 #loading time in ms
    time.sleep(1)
	#
	####################
	   #Filtering 
	####################
    t = time.time()
    pass_frequency=[0.6, 3]
    filt_sig_sample =_filtering(sig_sample[' ppg1'],fs,pass_frequency,f)
	#
    filtering_time = (time.time() - t)*1000 #filtering time in ms
    time.sleep(1)
	#
	####################
	   #Peak detection 
	####################
    t = time.time() 
    rpeaks = _Peak_detection(filt_sig_sample, fs, p)  
	#
    peak_detection_time = (time.time() - t)*1000 #peak detection time in ms
    time.sleep(1)
	#
	####################
	   #Feature extraction
	####################
    t = time.time()
    features1 = []
    features2 = []
    features3 = []
    features4 = []
    features5 = []
    features6 = []
    features7 = []
    features8 = []
        
    sp_mag = np.abs(np.fft.fft(filt_sig_sample))  ###one-dimensional discrete Fourier Transform
    freqs = np.fft.fftfreq(len(sp_mag))
    sp_mag_maxima_index = argrelmax(sp_mag)[0]
    fst = max(sp_mag[sp_mag_maxima_index])
    fst_in = np.where(sp_mag == fst) 
        
    df_features = pd.DataFrame(columns=['skewness', 'kurtosis', 'power'])
    heartpeak = _segmentation_heartCycle(filt_sig_sample,rpeaks)
    for i in range(heartpeak.shape[0]-1):
        heart_cycle =  filt_sig_sample[heartpeak[i]:heartpeak[i+1]]
        f_skew = stats.skew(heart_cycle)
        f_kurt = stats.kurtosis(heart_cycle)
        f_power = np.sum(np.power(heart_cycle,2))/heart_cycle.shape[0]
        df_features.loc[len(df_features)] = [f_skew,f_kurt,f_power]
        
    RR_interval = np.array([])       
    for i in range(rpeaks.shape[0]-1):
        RR_interval = np.append(RR_interval, np.abs(sig_sample['timestamp'][rpeaks[i+1]] - sig_sample['timestamp'][rpeaks[i]]))
        
    diff_nni = np.diff(RR_interval)
    features1.append(np.mean(RR_interval))
    features2.append(np.sqrt(np.mean(diff_nni ** 2)))       
    features3.append(np.std(RR_interval, ddof=1))   
    f_base = np.abs(freqs[fst_in][0]) * fs
    features4.append(f_base)
    sp_mag_base = np.abs(freqs[fst_in][0]) / len(filt_sig_sample)
    features5.append(sp_mag_base) 
    features6.append(_range(df_features['skewness']))
    features7.append(_range(df_features['kurtosis']))
    features8.append(_range(df_features['power']))

	
	#
	####################
	   #Normalize features
	####################
    features = np.zeros(8)*np.nan
    features[0] = (features1 - df_z['ave'][0])/df_z['std'][0]
    features[1] = (features2 - df_z['ave'][1])/df_z['std'][1]
    features[2] = (features3 - df_z['ave'][2])/df_z['std'][2]
    features[3] = (features4 - df_z['ave'][3])/df_z['std'][3]
    features[4] = (features5 - df_z['ave'][4])/df_z['std'][4]
    features[5] = (features6 - df_z['ave'][5])/df_z['std'][5]
    features[6] = (features7 - df_z['ave'][6])/df_z['std'][6]
    features[7] = (features8 - df_z['ave'][7])/df_z['std'][7]
    feature_extraction_time = (time.time() - t)*1000 #Feature extraction time in ms
    time.sleep(1)
	#
	####################
    	#ML - training
	####################
    t = time.time()
    y_pred = ML(features,clf)
	#
    ML_time = (time.time() - t)*1000 #machine learning time in ms
    time.sleep(1)
	#
    total_time = np.array([loading_time,filtering_time,peak_detection_time,feature_extraction_time,ML_time])
    filename_time_write = 'output_test/' + sys.argv[0] + '_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_' + sys.argv[4] + '_' + str(int(time.time())) + '.csv'
    np.savetxt(filename_time_write,total_time, fmt='%.3f')
	#
__main__()
