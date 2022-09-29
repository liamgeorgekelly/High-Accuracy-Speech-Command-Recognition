# 'OneDrive - University of Ottawa'/Documents/'My Documents'/'BrainStation Data Science Bootcamp'/Assignments/'Capstone Project'
# C:\Users\liamg\OneDrive - University of Ottawa\Documents\My Documents\BrainStation Data Science Bootcamp\Assignments\Capstone Project

# Import Packages:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.fft import fft, fftfreq
import os
import plotly.express as px
import pyaudio

# Initialize sections in the app:
header = st.container()
data_collection = st.container()
model_predict = st.container()
features = st.container()

# Initial Values
recorded = False # Voice has not been recorded yet.
s_rate = 16000 # Required sampling rate
audio_len = s_rate*3 # Length of audio recording


with header:
    st.title("Speech Recognition Project")
    st.markdown('#### XGBoost Model')
    st.markdown('By Liam Kelly')
    st.markdown("---")

with data_collection:

    st.subheader('Collect Voice Sample:')

    # Display table containing the available words for prediction:
    with st.expander("Click to View Words That the Model Can Recognize"):
        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.write('right')
        col_1.write('left')
        col_2.write('go')
        col_2.write('stop')
        col_3.write('up')
        col_3.write('down')
        col_4.write('yes')
        col_4.write('no')

    conf_thres = st.number_input(label='Minimum Confidence (%)',min_value=0.00, max_value=99.99,value = 99.90, format="%.2f")

    ################################
    ### Collect the Audio Sample ###
    ################################
    
    # Initialize audio stream:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=s_rate,
                input=True,
                frames_per_buffer=audio_len) 

    if st.button('Record Voice Sample'):
        st.write('Recording...')

        # Collect Data:
        x = np.frombuffer(stream.read(audio_len, exception_on_overflow=False),dtype=np.int16)

        # Convert the data to the normalized spectrum:
        x_hat = abs(x)/np.sum(abs(x))

        # Trim Data About Spectral Centroid:
        cent = int(np.sum(np.arange(len(x))*x_hat))
        if cent < int(s_rate/2):
            x = x[:s_rate]
        elif cent > len(x) - int(s_rate/2):
            x = x[int(2*s_rate):]
        else:
            x = x[cent-int(s_rate/2):cent + int(s_rate/2)]
        st.write('Recording Finished.')
        
        t = np.arange(len(x))/s_rate # Create time data.

        # Plot the collected audio sample
        fig = plt.figure()
        plt.plot(t,x)
        plt.title("Recorded Audio Sample")
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid()
        st.write(fig)

        recorded = True # Recording has been completed.


if recorded:
    with model_predict:
        ############################################
        ### Extract Feature Parameters from Data ###
        ############################################
        df = pd.DataFrame() # Initialize dataframe
        i=0 # Set the current dataframe index to 0.

        # Create array to ramp up the time domain signal from 0 to better represent a pulsed signal.
        mult = np.ones(len(x))
        mult[:1001] = np.linspace(0,1,1001)
        mult[-1001:] = np.linspace(1,0,1001)
        x = x*mult

        # Convert the data to the normalized spectrum:
        x_hat = abs(x)/np.sum(abs(x))
        
        ### Calculate Time-Domain Features ###
        # 1. Calculate the temporal bandwidth:
        temp_cent = np.sum(t*x_hat) # Temporal centroid
        temp_bw = np.sqrt(np.sum(((t - temp_cent)**2)*x_hat))
        df.loc[i, 'temporal_bandwidth'] = temp_bw
        
        # 2. Calculate the temporal skwedness
        temp_skew = np.sum(((t-temp_cent)**3)*x_hat)/(temp_bw**3)
        df.loc[i, 'temporal_skewness'] = temp_skew
        
        # 3. Calculate the temporal kurtosis:
        temp_kurt = np.sum(((t-temp_cent)**4)*x_hat)/(temp_bw**4)
        df.loc[i, 'temporal_kurtosis'] = temp_kurt
        
        # 4. Calculate the temporal irregularity:
        x_i = x_hat[:-1]
        x_i1 = x_hat[1:]
        temp_irreg = np.mean((x_i - x_i1)**2)/np.mean(x_hat**2)
        df.loc[i, 'temporal_irregularity'] =  temp_irreg
        
        
        ### Calculate Frequency Domain Features ###
        # Perform Fourier Transform (X), calculate frequency data (f) and normalized magnitude spectrum(X_hat)
        n_t = len(x) 
        X = 2/n_t*(abs(fft(x)))[:n_t//2]
        f = fftfreq(n_t, 1/s_rate)[:n_t//2]
        X_hat = abs(X)/np.sum(abs(X))
        
        # Connvert frequency to mel-scale:
        f_mel = 2595*np.log10(f/700 + 1)
        
        # 1. Calculate the spectral centroid:
        spec_cent = np.sum(f_mel*X_hat)
        df.loc[i, 'spectral_centroid'] = spec_cent
    
        # 2. Calculate the bandwidth of the spectrum:
        # Measures the weighted difference between spectral components and the spectral centroid.
        spec_bw = np.sqrt(np.sum(((f_mel - spec_cent)**2)*X_hat))
        df.loc[i, 'spectral_bandwidth'] = spec_bw
        
        # 3. Calculate the spectral skewdness:
        # Measures how the spectrum is skewed about the spectral centroid.
        spec_skew = np.sum(((f_mel-spec_cent)**3)*X_hat)/(spec_bw**3)
        df.loc[i, 'spectral_skewness'] = spec_skew
        
        # 4. Calculate the spectral kurtosis:
        # Measure of the peakiness of the spectrum
        spec_kurt = np.sum(((f_mel-spec_cent)**4)*X_hat)/(spec_bw**4)
        df.loc[i, 'spectral_kurtosis'] = spec_kurt
        
        # 5. Calculate the spectral flatness:
        # Is a measure of how 'white-noisy' as signal is. A Low value indicates a noisy signal, while a high value is indicative of tonal sounds 
        spec_flat = 10*np.log10(np.exp(np.log(X_hat).mean())/X_hat.mean())
        df.loc[i, 'spectral_flatness'] = spec_flat
        
        # 6. Calculate the spectral irregularity:
        X_i = X_hat[:-1]
        X_i1 = X_hat[1:]
        spec_irreg = np.mean((X_i - X_i1)**2)/np.mean(X_hat**2)
        df.loc[i, 'spectral_irregularity'] =  spec_irreg
        
        # 7. Calculate the spectral roll-off:
        tot_energy = np.sum(X_hat**2)
        indices = np.where(np.cumsum(X_hat**2) > 0.95*tot_energy)
        df.loc[i, 'spectral_rolloff'] = min(f_mel[indices])  

        # Save the dataframe post feature extraction:
        df_extract = df

        ###########################
        ### Data Transformation ###
        ###########################

        # Clean Data:
        df['temporal_bandwidth'] = np.power(df['temporal_bandwidth'],1/6)
        df['temporal_kurtosis'] = np.log(df['temporal_kurtosis'])
        df['temporal_irregularity'] = np.power(df['temporal_irregularity'],1/4)
        df['spectral_centroid'] = np.power(abs(df['spectral_centroid']),1/4)
        df['spectral_bandwidth'] = np.power(np.reciprocal(df['spectral_bandwidth']),1/8)
        df['spectral_kurtosis'] = np.power(np.reciprocal(df['spectral_kurtosis']),1/2)
        df['spectral_flatness'] = np.power(abs(df['spectral_flatness']),1/4)
        df['spectral_irregularity'] = np.log(df['spectral_irregularity'])

        ########################
        ### Fit to the Model ###
        ########################
        
        X = df

        # Load the models:
        xgb_model = pickle.load(open('models/xgb_model.sav', 'rb'))
        le = pickle.load(open('models/label_encoder.sav', 'rb'))

        # Use the model to predict the spoken word:
        pred_val = le.inverse_transform(xgb_model.predict(X))[0]
        pred_prob = xgb_model.predict_proba(X)*100
        conf = pred_prob.max()

        # Print the outcome:
        # If the confidence is above the specified threshold, then success:
        if conf > conf_thres:

            st.header('Success! You said: %s' % pred_val)
            st.write('Confidence = %.1f %%' % conf) 

        # If the confidence is below the specified threshold, then failure:
        else:

            st.header('Failure! We thought you said: %s, but are uncertain' % pred_val)
            st.write('Confidence = %.1f %%' % conf) 

    with features:
        
        with st.expander('Click to View Additional Details'):
            st.header('Confidence Values')
            # Create a variable of all possible words:
            y_val = le.inverse_transform(np.arange(8))

            # Graph a polar plot of the models confidence in each word:
            st.write('Polar Plot of Confidence Values for Each Word:')
            fig_polar = px.line_polar(r=pred_prob[0], theta=y_val, line_close=True)
            fig_polar.update_traces(fill='toself')
            st.write(fig_polar)

            # Bar plot of the model's confidence in each word:
            fig_bar = px.bar(x=y_val, y=pred_prob[0])
            fig_bar.update_layout(title='Bar Plot of Confidence Values for Each Word',
                      xaxis_title='Spoken Word', yaxis_title='Confidence (%)')
            st.write(fig_bar)

            
            # Display the original Audio Sample
            st.header('Calculated Audio Features')

            for col in  df_extract.columns:
                st.write('%s: %.3f' % (col.replace('_',' '), df_extract.loc[0,col]))

