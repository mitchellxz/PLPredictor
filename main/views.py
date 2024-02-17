from django.shortcuts import render
from .models import Match
from django.db import connection
from .prediction import getResults
import pandas as pd
import os
from django.conf import settings
from .train_test_model import trainingTesting



# Create your views here.

def index(request):
    matches = Match.objects.all()
    return render(request, "main/index.html", {'matches': matches})

def home(request):
    with connection.cursor() as cursor:
        cursor.execute("""
                       SELECT REPLACE(substring(column_name FROM 10), '_', ' ') AS modified_column_name
                        FROM information_schema.columns
                        WHERE table_name = 'main_match'
                            AND column_name LIKE 'HomeTeam%';

                        """
                       )
        columns = [row[0] for row in cursor.fetchall()]
    
    return render(request, 'main/home.html', {'columns': columns})

def results(request):
    home_team = None
    away_team = None
    if request.method == 'POST':
        home_team = request.POST.get('home_team_select')
        away_team = request.POST.get('away_team_select')

    if home_team == away_team:
        error_message = "Home and away teams must be different."
        return render(request, 'main/results.html', {'error_message': error_message})
    
    csv_file_features = os.path.join(settings.STATIC_ROOT, 'main/csv/featureslist.csv')
    csv_file_label = os.path.join(settings.STATIC_ROOT, 'main/csv/labellist.csv')

    features = pd.read_csv(csv_file_features)
    label = pd.read_csv(csv_file_label)

    model = trainingTesting(label, features)

    expected_outcome = getResults(home_team, away_team, features, model)
    
    
    
    return render(request, "main/results.html", {'home_team':home_team, 'away_team':away_team, 'expected_outcome':expected_outcome})