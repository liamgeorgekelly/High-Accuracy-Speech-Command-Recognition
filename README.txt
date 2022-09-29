###########################################################################
### High Accuracy Speech Recognition for Industrial Machinery Operation ###
###########################################################################

By: Liam Kelly
September 26th 2022

###########################################################################

############
Introduction
############

Speech recognition is a powerful tool that can be used as an alternative method for providing user inputs. In industrial settings, traditional methods of user input (e.g., buttons, levers, pedals) can be difficult to operate when wearing protective clothing, or if workers are physically impaired. By developing a highly accurate speech recognition model, voice commands can be used to replace traditional user inputs in areas where they are cumbersome. 

Existing benchmark models have demonstrated a keyword spotting accuracy of over 98% on the speech command dataset used in this report. While it’s unlikely to exceed that accuracy here, the goal of this project is to investigate the use of various machine learning models and to better understand their advantages and limitations.

############################
Files and Order of Operation
############################

0. capstone_env.txt

	File which can be used to create the virtual environment used in this project.

### Traditional Classification Models ###

1. extraction_of_audio_features.ipynb
	
	Applies various functions to audio waveforms to extract its acoustic properties 		and store them in a dataframe.

2. EDA_of_audio_data.ipynb

	Perform EDA on the extracted audio features, perform feature transformation to 			improve normality.

3. classification_models.ipynb

	Optimize and evaluate the performance of various classification models.

### CNN Model ###

4. spectrogram_maker.ipynb
	
	Creates spectrograms from audio waveforms.

5. CNN_model.ipynb 

	Performs CNN modelling on the created spectrograms.

### Streamlit Apps ###

6. baseline_app.py

	Streamlit app based on XGBoost model.

7. CNN_app.py

	Streamlit app based on CNN model.
	
