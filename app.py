"""Starter Plotly Dash app for the SpaceX capstone."""
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

# Put the official course dataset at data/dataset_part_2.csv
df = pd.read_csv('data/dataset_part_2.csv')
app = Dash(__name__)
app.layout = html.Div([
    html.H1('SpaceX Falcon 9 Landing Analysis'),
    dcc.Graph(figure=px.pie(df, names='Class', title='Landing outcome distribution')),
    dcc.Graph(figure=px.scatter(df, x='PayloadMass', y='Class', color='LaunchSite', hover_data=['Orbit'], title='Payload mass vs landing outcome')),
])
if __name__ == '__main__':
    app.run(debug=True)
