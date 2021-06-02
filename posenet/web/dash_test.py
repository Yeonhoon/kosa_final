import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# layout ---------------------------------------------------------
app.layout = html.Div(id='dash-container',
    children=[
    html.H1(children = 'Hello Dash'),
    html.Div(children="""
        Dash Test
    """),

    dcc.Graph(
        id = 'example-graph',
        figure = {
            'data': [
                {'x': [1,2,3],'y':[4,1,2], 'type':'bar','name':'SF'},
                {'x': [1,2,3],'y':[4,1,2], 'type':'line','name':'line'},
            ],
            'layout': {
                'title': 'Data Visualization'
            }

        }
    )

])

if __name__ == '__main__':
    app.run_server(debug=True)
