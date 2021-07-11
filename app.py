import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import pandas as pd
import addPred

app = dash.Dash(__name__)
server= app.server
# color pallette
cnf = '#393e46'  # confirmed - grey
dth = '#ff2e63'  # death - red
rec = '#21bf73'  # recovered - cyan
act = '#fe9801'  # active case - yellow
blk = '#000000'  # black
plp = '#663399'  # purple



#prediction
addPred.addRecord()
data=pd.read_csv('predictionCovid.csv')
figPred = go.Figure()
figPred.add_trace(go.Scatter(x=data.Date, y=data.Predicted,
                    mode='lines+markers',
                    name='Predicted Cases'))
figPred.add_trace(go.Scatter(x=data.Date, y=data.Actual,
                    mode='lines+markers',
                    name='Actual Cases'))




app.layout = html.Div(children=[
   
    html.Div(children='''PREDICTION OF CONFIRMED CASES'''), dcc.Graph(id='figPred',figure=figPred)
   
])

if __name__ == '__main__':
    app.run_server(debug=False,host='0.0.0.0')


