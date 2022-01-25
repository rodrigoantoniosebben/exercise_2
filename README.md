# Exercice 2  - Disaster Response Pipeline Project
Udacity Data Scientist Nanodegree Project.

## Table of contents
1. [Introduction](#introduction)
2. [Project Motivation](#motivation)
3. [How to use](#how_to_use)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
   
## Introduction  <a name="introduction"></a>
This repository solves the second exercise of the Data Science Nanodegree from Udacity called Disaster Response Pipeline Project.

The files of the project has the following architecture.

* app

| - template

| |- master.html  # main page of web app

| |- go.html  # classification result page of web 

|- run.py  # Flask file that runs app

* data

|- disaster_categories.csv  # data to process 

|- disaster_messages.csv  # data to process

|- process_data.py

|- InsertDatabaseName.db   # database to save clean data to

* models

|- train_classifier.py

|- classifier.pkl  # saved model 


## Project Motivation <a name="motivation"></a>
A disaster recovery situation needs a fast and accurate response. One way to produce good actions that will result in qualified responses is to help the first line to understand what kind of need is being requested to correctly dispatch the team/help needed. 

This project has the goal to create a server that helps the first line support to understand and categorize a text received from a disaster situation. The categories available will be used to bring efficiency in the responses to the disaster making it possible for the support line to request actions from the correct people. 


## How to use <a name="how_to_use"></a>
1. Install dependencies.

* re
* nltk
* numpy
* pandas
* pickle
* sqlite3
* sklearn
* sqlalchemy

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results <a name="result"></a>

The result is splitted into 3 parts

### Part 1. ETL Pipeline

At the end of the ETL pipeline, a sql file will be created with the data extrated and transofrmed from message and categories.

### Part 2. ML Pipeline

At the end of the ML Pipeline a pkl file will be generated. This file contains the result of the Machine Learning pipeline and will be used by  the server to check furhter messages.

### Part 3. Server

After running the server with he pkl file, it's possible to run a message and check which categories that message triggers. 

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

The author of this project is Rodrigo Sabben

The code is licensed under MIT License. 