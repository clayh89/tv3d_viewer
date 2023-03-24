import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

df = pd.read_csv('plot.ly5.csv').T
df.columns = [3, 12, 35, 100]
df = df.reset_index()

rename_dict = {'P': 'power', 'L': 'luminance', 'A': 'area'}
df['measurement'] = df['index'].apply(lambda x: rename_dict[x[-1]])
df['tv'] = df['index'].apply(lambda x: x[:-2])

df = df.drop('index', axis=1)

mdf = df.melt(id_vars=['measurement', 'tv'], var_name='lux')

fdf = mdf.set_index(['tv', 'lux']).pivot(columns='measurement')

fdf.columns = list(fdf.columns.levels[1])
brand_rename = {'lg': 'LG', 'so': 'Sony', 'sa': 'Samsung', 'vi': 'Vizio'}

fdf = fdf.reset_index()
fdf['brand'] = fdf['tv'].apply(lambda x: brand_rename[x[:2].lower()])

app = dash.Dash()
app.config.suppress_callback_exceptions = True
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])
# app.layout = html.Div(children=[
#
#     html.B('Grey Surface Equation   '),
#     dcc.Input(
#         id='surface1-equation',
#         type='text',
#         value='0.2175*(x*y)**0.5661'
#     ),
#     html.Br(),
#     html.Br(),
#     html.B('Blue Surface Equation   '),
#     dcc.Input(
#         id='surface2-equation',
#         type='text',
#         value='.0526*x+12.911'
#     ),
#     html.Div(id='abc-on-plot'),
#     html.Br(),
#     html.Br(),
#     html.Br(),
#     html.Br(),
#     html.B('Grey Surface Equation   '),
#     dcc.Input(
#         id='surface3-equation',
#         type='text',
#         value='(0.11841843059765*(x*y)**0.632781035697225)*0.95'
#     ),
#     html.Br(),
#     html.Br(),
#     html.B('Blue Surface Equation   '),
#     dcc.Input(
#         id='surface4-equation',
#         type='text',
#         value='(0.0831*x+13.635)*1.2'
#     ),
#     html.Div(id='abc-off-plot'),
# ])

page_1_layout = html.Div(children=[
    html.B('Grey Surface Equation   '),
    dcc.Input(
       id='surface1-equation',
       type='text',
       value='0.2175*(x*y)**0.5661'
    ),
    html.Br(),
    html.Br(),
    html.B('Blue Surface Equation   '),
    dcc.Input(
       id='surface2-equation',
       type='text',
       value='.0526*x+12.911'
    ),
    html.Div(id='abc-on-plot')
])

page_2_layout = html.Div(children=[
    html.B('Grey Surface Equation   '),
    dcc.Input(
        id='surface3-equation',
        type='text',
        value='(0.11841843059765*(x*y)**0.632781035697225)*0.95'
    ),
    html.Br(),
    html.Br(),
    html.B('Blue Surface Equation   '),
    dcc.Input(
        id='surface4-equation',
        type='text',
        value='(0.0831*x+13.635)*1.2'
    ),
    html.Div(id='abc-off-plot'),
])


@app.callback(
    Output('abc-on-plot', 'children'),
    [Input('surface1-equation', 'value'), Input('surface2-equation', 'value')])
def abc_on_plot(eq1, eq2):
    fig = go.Figure()
    color_dict = {'Sony': 'green', 'LG': 'blue', 'Samsung': 'red', 'Vizio': 'yellow'}
    for tv in fdf.tv.unique():
        tv_df = fdf[fdf['tv'] == tv].sort_values(by='lux')
        brand = tv_df['brand'].iloc[0]
        color = color_dict[brand]
        marker = {'color': color, 'size': 4}
        text = tv_df['lux'].astype(str).to_list()
        hovertemplate = 'lux: %{text}<br>' + 'sq inches: %{x:.0f}' + '<br>nits: %{y:.0f}' + '<br>watts:  %{z:.0f}'
        scatter = go.Scatter3d(x=tv_df['area'], y=tv_df['luminance'], z=tv_df['power'],
                               name=tv, marker=marker, text=text, hovertemplate=hovertemplate,
                               legendgroup=brand, showlegend=False)
        fig.add_trace(scatter)

    for brand, color in color_dict.items():
        marker = {'color': color, 'size': 0}
        hidden_scatter = go.Scatter3d(x=np.arange(2), y=np.arange(2), z=np.arange(2), name=brand, marker=marker,
                                      legendgroup=brand)
        fig.add_trace(hidden_scatter)

    xx = np.linspace(0, 3200, 10)
    yy = np.linspace(0, 125, 10)
    x, y = np.meshgrid(xx, yy)
    z1 = eval(eq1)
    z2 = eval(eq2)
    surface = go.Surface(x=x, y=y, z=z1, surfacecolor=np.full(z1.shape, -100), colorscale='Greys', showlegend=False,
                         showscale=False, hoverinfo='skip', opacity=.87)
    surface2 = go.Surface(x=x, y=y, z=z2, surfacecolor=np.full(z2.shape, -100), colorscale='Blues', showlegend=False,
                          showscale=False, hoverinfo='skip', opacity=.87)

    fig.add_trace(surface)

    fig.add_trace(surface2)

    fig.update_layout(scene=dict(
        xaxis_title='Area',
        yaxis_title='Luminance',
        zaxis_title='Power'),
    )
    fig.update_layout(height=900)
    return dcc.Graph(figure=fig)

@app.callback(
    Output('abc-off-plot', 'children'),
    [Input('surface3-equation', 'value'), Input('surface4-equation', 'value')])
def abc_off_plot(eq1, eq2):
    path = r"plot.ly2.csv"
    df = pd.read_csv(path)

    df = df.iloc[:, :4]

    df.columns = ['tv', 'power', 'area', 'luminance']
    brand_rename = {'lg': 'LG', 'so': 'Sony', 'sa': 'Samsung', 'vi': 'Vizio'}
    df['brand'] = df['tv'].apply(lambda x: brand_rename[x[:2].lower()])

    color_dict = {'Sony': 'green', 'LG':'blue', 'Samsung': 'red', 'Vizio': 'purple'}

    fig = go.Figure()

    for brand in df['brand'].unique():
        brand_df = df[df['brand']==brand]
        color = color_dict[brand_df['brand'].iloc[0]]
        marker = {'color': color, 'size': 4}
        text = brand_df['tv'].to_list()
        hovertemplate = '%{text}<br>' + 'sq. in.: %{x:.0f}' + '<br>nits: %{y:.0f}' + '<br>watts:  %{z:.0f}'
        scatter = go.Scatter3d(x=brand_df['area'], y=brand_df['luminance'], z=brand_df['power'],
                               marker=marker, mode='markers', name=brand,
                               hovertemplate=hovertemplate, text=text)
        fig.add_trace(scatter)

    max_area = df['area'].max()
    max_lum = df['luminance'].max()
    xx, yy = np.linspace(0, max_area, 10), np.linspace(0, max_lum, 10)
    x, y = np.meshgrid(xx, yy)

    z1 = eval(eq1)
    z2 = eval(eq2)

    surface1 = go.Surface(x=x, y=y, z=z1,
                         surfacecolor=np.full(z1.shape, -100), colorscale='Greys', opacity=.87,
                         showlegend=False, showscale=False, hoverinfo='skip')
    surface2 = go.Surface(x=x, y=y, z=z2,
                         surfacecolor=np.full(z2.shape, -100), colorscale='Blues', opacity=.87,
                         showlegend=False, showscale=False, hoverinfo='skip')

    fig.add_trace(surface1)
    fig.add_trace(surface2)
    fig.update_layout(scene = dict(
                        xaxis_title='Area',
                        yaxis_title='Luminance',
                        zaxis_title='Power'),
                     )
    fig.update_layout(height=900)
    return dcc.Graph(figure=fig)





@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)