from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# Configuration de la base de données
DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(DATABASE_URL)

# Charger les données
def load_data():
    df = pd.read_sql("SELECT * FROM predictions", engine)
    return df

# Initialiser l'application Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Tableau de Bord des Performances"),
    dcc.Graph(id='performance-graph'),
    dcc.Graph(id='time-series-graph'),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # en millisecondes (1 minute)
        n_intervals=0
    )
])

@app.callback(
    Output('performance-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    df = load_data()
    fig = px.histogram(df, x='spam', title='Distribution des Prédictions (Spam vs Non-Spam)')
    return fig

@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_timeseries(n):
    df = load_data()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fig = px.line(df, x='timestamp', y='probability', title='Évolution des probabilités de spam')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
