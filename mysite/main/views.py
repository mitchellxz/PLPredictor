from django.shortcuts import render
from django.http import HttpResponse
from .models import Match
from django.db import connection
from .prediction import team_names  # Import the main function
import pandas as pd


# Create your views here.

def index(response):
    matches = Match.objects.all()
    return render(response, "main/base.html", {'matches': matches})

def home(response):
    matches = Match.objects.all()
    return render(response, "main/home.html", {'matches': matches})

def display_columns(response):
    with connection.cursor() as cursor:
        cursor.execute("""
                       SELECT REPLACE(substring(column_name FROM 10), '_', ' ') AS modified_column_name
                        FROM information_schema.columns
                        WHERE table_name = 'main_match'
                            AND column_name LIKE 'HomeTeam%';

                        """
                       )
        columns = [row[0] for row in cursor.fetchall()]
    
    return render(response, 'main/display_columns.html', {'columns': columns})

def results(response):
    home_team = None
    away_team = None
    if response.method == 'POST':
        home_team = response.POST.get('home_team_select')
        away_team = response.POST.get('away_team_select')

    if home_team == away_team:
        error_message = "Home and away teams must be different."
        return render(response, 'main/results.html', {'error_message': error_message})
    
    team_names(home_team, away_team)
    
    
    
    return render(response, "main/results.html", {'home_team':home_team, 'away_team':away_team})