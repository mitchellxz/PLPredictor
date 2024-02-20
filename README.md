# PL Predictor
## Description
This repository contains an analysis of machine learning models applied to predict outcomes in Premier League (PL) football matches.

## Dataset
The dataset used for this project is sourced from [Football Data](https://www.football-data.co.uk/englandm.php). The preprocessed version of the dataset can be found in the provided directory of CSV files.

## Models and Accuracy
This section contains an analysis of the performance of two machine learning models for predicting outcomes in Premier League (PL) football matches: Multinomial Naïve Bayes and Random Forest.
### Multinomial Naïve Bayes
![confusion matrix of multinomial naïve bayes](/analysis_images/conf_matrix_multinomialnb.png)

Overall Accuracy: 80%
### Random Forest
![confusion matrix of random forest](/analysis_images/conf_matrix_random_forest.png)

Overall Accuracy: 89%

## Results
The Random Forest model achieved an impressive overall accuracy of 89%, outperforming the Multinomial Naïve Bayes model. However, it is essential to note that the Random Forest model displayed a notable bias towards predicting draws. Conversely, the Multinomial Naïve Bayes model demonstrated a tendency to favor home teams, with fewer draws predicted. This observation aligns with the typical outcome distribution in football matches, where home teams often secure more wins.

## Installation (Local)
### Clone Project
```bash
git clone (https://github.com/mitchellxz/PLPredictor.git)
```
### Install Dependencies
```bash
python -m venv venv
# cd into \venv\Scripts and enter activate
pip install -r requirements.txt
```
### Create Database
```bash
# Assuming you have installed Postgresql
psql -U your_username
# Enter password
CREATE DATABASE database_name;
```
### Configure settings.py
```python
DATABASES = {
  'default': {
    'ENGINE': 'django.db.backends.postgresql',
    'NAME': 'database_name',
    'USER': 'username',
    'PASSWORD': 'password',
    'HOST': 'localhost,
  }
}
```
### Migrate Project To Database
```bash
python manage.py makemigrations
python manage.py migrate
```
### Run Server
```bash
python manage.py runserver
```
## License

[MIT](https://choosealicense.com/licenses/mit/)
