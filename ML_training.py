import pandas as pd
import numpy as np
import glob
from joblib import dump
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from biosppy.signals.tools import filter_signal
import scipy.signal as signal
from pathlib import Path
from scipy.signal import argrelmax
import heartpy as hp
from sklearn.cluster import KMeans
from heartpy.datautils import rolling_mean
from scipy import stats

Version = '1' #select version
f = 2 #filters: 1 is firwin, 2 is butterworth, 3 is bessel, and 4 is elliptic
p = 1 #peak detection: 1 is hamilton, 2 is Slope Sum Function, 3 is Christov, 4 is Engelse and Zeelenberg
#m = 4 #machine learning: 1 is SVM, 2 is ANN, 3 is KNN, 4 is RFC

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


def ML_train(df,n):
    X = df[['f1','f2','f3','f4','f5','f6','f7','f8']]
    filenameLabel_read = 'myData'+Version+'/trainingDataLabel.csv'
    label = pd.read_csv(filenameLabel_read)
    y = label['label']
    X.dropna(inplace=True)
    X.reset_index(drop=True, inplace=True)
    X = X.to_numpy()
    if n == 1:
        clf = SVC(gamma=0.1,kernel='linear', C = 10)
        clf.fit(X, y)
        modelName = 'SVM_L'        
    elif n == 2:
        clf = SVC(gamma=0.1,kernel='rbf', C = 10)
        clf.fit(X, y)
        modelName = 'SVM_RBF'
    elif n == 3:	
        clf = MLPClassifier(hidden_layer_sizes=(8,8,),activation='relu',solver='adam',max_iter=500,learning_rate='invscaling',alpha=0.001)
        clf.fit(X, y)
        modelName = 'ANN'
    elif n == 4:
        clf = KMeans(n_clusters=2, random_state=0)
        clf.fit(X)
        modelName = 'KMeans'
        
    print('classifier = ' + modelName)
    dump(clf, 'myModel'+Version+'/' + modelName + '_' + 'F' + str(f) + '_' + 'P' + str(p) + '.joblib') 

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
	#Read file 
	####################
    filename_read = glob.glob('Train_dataset/T-*.csv')
    fs = 25
	#   
    features1 = []
    features2 = []
    features3 = []
    features4 = []
    features5 = []
    features6 = []
    features7 = []
    features8 = []
    for filename in filename_read:
        sigfile = pd.read_csv(filename, delimiter=',')  
		####################
    		#Filtering 
		####################
        pass_frequency=[0.6, 3]
        filt_sig_sample =_filtering(sigfile[' ppg1'].values.reshape(1,-1)[0],fs,pass_frequency,f)
        ####################### 
            #Peak Detection
        ####################### 
        rpeaks = _Peak_detection(filt_sig_sample, fs, p)        
        ####################
	        #Feature Extraction
	    ####################    
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
            RR_interval = np.append(RR_interval, np.abs(sigfile['timestamp'][rpeaks[i+1]] - sigfile['timestamp'][rpeaks[i]]))
        
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
    
    df_data = list(zip(*[features1, features2, features3, features4, features5, features6, features7, features8]))
    df = pd.DataFrame(df_data, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'])
    df = df.dropna()
    
	# 
    ####################
    	#Normalize features
	####################def Normalized():        
    df_z = pd.DataFrame(columns=['ave','std'])
    df_z['ave'] = ([np.mean(df['f1']),np.mean(df['f2']),np.mean(df['f3']),np.mean(df['f4']),np.mean(df['f5']),np.mean(df['f6']),np.mean(df['f7']),np.mean(df['f8'])])
    df_z['std'] = ([np.std(df['f1']),np.std(df['f2']),np.std(df['f3']),np.std(df['f4']),np.std(df['f5']),np.std(df['f6']),np.std(df['f7']),np.std(df['f8'])])
    Path('myModel'+Version).mkdir(parents=True, exist_ok=True)
    filename_zscore_write = 'myModel'+Version+'/zscore.csv'
    df_z.to_csv(filename_zscore_write,sep=',',index=False)
    df['f1'] = (df['f1'] - df_z['ave'][0])/df_z['std'][0]
    df['f2'] = (df['f2'] - df_z['ave'][1])/df_z['std'][1]
    df['f3'] = (df['f3'] - df_z['ave'][2])/df_z['std'][2]
    df['f4'] = (df['f4'] - df_z['ave'][3])/df_z['std'][3]
    df['f5'] = (df['f5'] - df_z['ave'][4])/df_z['std'][4]
    df['f6'] = (df['f6'] - df_z['ave'][5])/df_z['std'][5]
    df['f7'] = (df['f7'] - df_z['ave'][6])/df_z['std'][6]
    df['f8'] = (df['f8'] - df_z['ave'][7])/df_z['std'][7]
    
	#
	####################
    	#ML - training
	####################
    for m in range(1,5):
        ML_train(df,m)

__main__()
