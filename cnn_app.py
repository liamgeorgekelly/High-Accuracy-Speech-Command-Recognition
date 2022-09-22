# 'OneDrive - University of Ottawa'/Documents/'My Documents'/'BrainStation Data Science Bootcamp'/Assignments/'Capstone Project'
# C:\Users\liamg\OneDrive - University of Ottawa\Documents\My Documents\BrainStation Data Science Bootcamp\Assignments\Capstone Project

# Import Packages:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.io import wavfile 
from scipy.fft import fft, fftfreq
import librosa
import os
import pyaudio


# Initialize sections in the app:
header = st.container()
data_collection = st.container()
model_predict = st.container()
additional_info = st.container()

# Initial Values
recorded = False # Voice has not been recorded yet.
s_rate = 16000 # Required sampling rate
audio_len = s_rate*3 # Length of audio recording


with header:
    st.title("Speech Recognition Project")
    st.markdown('#### CNN Model')
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
        
        recorded = True # Recording has been completed.


if recorded:
    with model_predict:
        #################################################
        ### Convert the Recorded Audio to Spectrogram ###
        #################################################

        # Define the Spectrogram making function:
        def my_spectrogram(x, s_rate, window_s = None, n_overlap = None, n_scale = 10, pad = True, pad_l = 10, trend = True, scale = 'log'):
            from scipy.fft import fft, fftfreq
            import numpy as np
            from scipy.signal import detrend
            
            # Initialize parameters:
            n_t =len(x) # Obtain the length of the time domain signal
            if window_s == None:
                window_s = int(n_t/50) # Choose the size of the window to perform Fourier transforms over
                print(window_s)
            if n_overlap == None:
                n_overlap = int(window_s*1/8) # chose the number of points to overlap between segments.
            step_size = window_s - n_overlap # Number of data points between FFT steps.
            n_FFTs = int((n_t-window_s)/step_size) # Total number of FFTs.
            
            # Perform FFT over each slice of time domain data:
            if pad == True:
                spect = np.zeros([int((window_s + 2*pad_l)/2), n_FFTs]) # Initialize the spectrogram array.
            if pad == False:
                spect = np.zeros([int(window_s/2), n_FFTs]) # Initialize the spectrogram array.
            for i in range(n_FFTs):
                x_c = x[i*step_size: i*step_size + window_s] # Slice out data for the current FFT.
                
                # Detrend the data to prevent computational artifacts near low frequencies:
                if trend == True:
                    x_c = detrend(x_c)
                
                # Linear scale first and last values to prevent delta function:
                x_c[:n_scale] = x_c[:n_scale]*np.linspace(0,1,n_scale)
                x_c[-n_scale:] = x_c[-n_scale:]*np.linspace(1,0,n_scale)
                
                if pad == True:
                    x_c = np.pad(x_c, pad_l, mode = 'constant') 
                n = len(x_c)
                
                # Perform the FFT on the current slice
                X_c = 2/n * (np.abs(fft(x_c)[:n//2]))
                if scale == 'log':
                    X_c = 10*np.log10(X_c)
                spect[:,i] = X_c
                
            return spect

        # Calculate the spectrogram:
        spec = my_spectrogram(x, s_rate, window_s = 250, n_scale = 50)

        # Create array to contain the spectrogram for fittign
        X = np.empty((1, 135, 71))  

        if np.isinf(spec).sum() > 0:
            X = np.delete(X, 0, 0) # If infinite values, drop the row from the X array.
        
        # If no infinite values, continue processing data:
        else:
            # Scale the data so the values are between 0 and 1:
            spec = spec - spec.min()
            spec = spec/spec.max()
            
            # Check that the data has the correct shape:
            if spec.shape != (135,71):
                
                # If the data has less rows:
                if spec.shape[1] < 71:
                    X[0,:,:] = 0 # Write a baseline value of the minimum value of the array.
                    X[0,:spec.shape[0], :spec.shape[1]] = spec # Overwrite the first values of the X array with the data from the spectrogram.
                
                # If the data has more rows:
                else:
                    X[0,:,:] = spec[:135,:71] # Take only the first 135 rows and 71 columns from the data.
                    
            # If data has the correct number of rows:
            else:
                X[0,:,:] = spec # Save the spectrogram data to the appropriate row in the X array.

        # Plot the spectrogram:
        fig = plt.figure()
        plt.imshow(X[0], cmap='viridis', interpolation='nearest', aspect= 'auto')
        plt.colorbar()
        plt.title('Spectrogram of Recorded Audio')
        st.write(fig)

        # Reshape for Keras model types:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        ########################
        ### Fit to the Model ###
        ########################
        
        # from tensorflow import keras
        import tensorflow
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import MaxPooling2D
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import BatchNormalization


        # Initialize the model:
        CNN_model = Sequential()
        num_classes = 8
        # Create simple CNN model architecture with Pooling for dimensionality reduction 
        # and Dropout to reduce overfitting
        CNN_model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation = 'relu', input_shape = (135, 71, 1)))
        CNN_model.add(MaxPooling2D(pool_size=(2, 2)))
        CNN_model.add(BatchNormalization())
        CNN_model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation = 'relu'))
        CNN_model.add(MaxPooling2D(pool_size=(2, 2)))
        CNN_model.add(BatchNormalization())
        CNN_model.add(Conv2D(filters = 128, kernel_size=(3, 3), activation = 'relu'))
        CNN_model.add(MaxPooling2D(pool_size=(2, 2)))
        CNN_model.add(BatchNormalization())
        CNN_model.add(Conv2D(filters = 256, kernel_size=(3, 3), activation = 'relu'))
        CNN_model.add(MaxPooling2D(pool_size=(2, 2)))
        CNN_model.add(BatchNormalization())
        CNN_model.add(Dropout(0.25))

        # Flatten the output of our convolutional layers
        CNN_model.add(Flatten())

        # Add dense layers
        CNN_model.add(Dense(512, activation='relu'))
        CNN_model.add(BatchNormalization())
        CNN_model.add(Dropout(0.5))
        CNN_model.add(Dense(64, activation = 'relu'))
        CNN_model.add(BatchNormalization())
        CNN_model.add(Dense(num_classes, activation='softmax'))

        # Load model and transformers:
        CNN_model.load_weights('models/CNN_weights')
        le = pickle.load(open('models/label_encoder.sav', 'rb'))

    
        # Predict values and record confidence:
        pred = CNN_model.predict(X)*100
        conf = pred.max()
        pred_i = np.array([np.argmax(pred)])
        pred_val = le.inverse_transform(pred_i)

        # Print the results:
        st.header('Predicted Value: %s' % pred_val[0])
        st.subheader('Confidence = %.3f %%' % conf) 
        st.write(' ')

    with additional_info:
        
        with st.expander('Click to View Additional Details'):
            
            # Create a table which contains the model's confidence for all words:
            y_val = le.inverse_transform(np.arange(8))
            df_pred = pd.DataFrame(data = pred, columns = y_val).round(3)
            df_pred.index = ['Confidence (%)']

            # Display the table:
            st.write('Confidence Values for Each Word:')
            st.write(df_pred)

            # Display the original Audio Sample
            # Plot the collected audio sample
            fig_og = plt.figure()
            plt.plot(t,x)
            plt.title("Recorded Audio Sample")
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.grid()
            st.write(fig_og)