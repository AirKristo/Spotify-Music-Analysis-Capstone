# Spotify Music Analysis Capstone

This project analyzes a dataset of 52,000 Spotify songs to uncover insights into what makes music popular. It involves data preprocessing, exploratory data analysis, and the application of machine learning models to predict various aspects of songs, such as their popularity and energy levels. The goal is to

# Objective

This capstone project is designed to integrate the various concepts and techniques learned throughout the course. The task is to be a Data Scientist at Spotify that is given a dataset and is looking to understand what factors go into making a song popular and the what audio features may up a genre. The primary goal of this analysis is to explore and identify key insights from a dataset of 52,000 songs. 
- Want to determine the features that influence a song's popularity
- Create figures to display results of analysis
- Clean and process data
- Analyze the audio characteristics that differentiate music genres.

# Datasets

The datasets used in this project were provided in the Intro to Data Science course at New York University, it is not included in this repo and would have to be obtained through the course.

# Dataset Description
The dataset used in this analysis consists of 52,000 songs with various attributes, including:
- **Popularity**: An integer value from 0 to 100 representing how popular a song is on Spotify.
- **Duration**: The length of the song in milliseconds.
- **Audio Features**: Attributes like danceability, energy, loudness, and tempo, among others.
- **Genre**: The genre of the song, with 1,000 songs sampled from each genre.
- **SongNumber**: The track ID of the song
- **Track Name**: Name of track corresponding to the track ID

The project also has a file that contains feedback in the form of star ratings from 10,000 users on 5,000 songs. Each row corresponds to a user and each column is a song, so the first 5,000 songs of the 52,000 song dataset in the same order. Ratings go from 0 to 4, where 0 is low and 4 is highly rated.

# Usage

To run the analysis, open the Jupyter notebook `Spotify Capstone.ipynb` and execute the cells sequentially. The notebook is divided into sections corresponding to different analysis tasks.

# Approach
This project is structured around answering 10 specific questions related to the dataset. Each question is addressed with a combination of statistical analysis, visualizations, and machine learning models, as appropriate. The analysis includes data preprocessing steps such as handling missing data, dealing with duplicates, and feature engineering. A bonus extra credit is included at the end. 
