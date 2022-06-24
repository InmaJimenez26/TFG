import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from PIL import Image
import plotly.express as px
import requests
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import lime.lime_image as li
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import tensorflow as tf
from keras.applications import inception_resnet_v2 as inc_net

torch.hub.set_dir("./")

def format_label(label):
    label = " ".join(label.split()[1:])
    label = ",".join(label.split(",")[:3])
    return label


def Header(name, app):
    title = html.H2(name, style={"margin-top": 15}) # pongo H2 que es mas grande
    logo = html.Img(
        src=app.get_asset_url("logo-uma.jpg"), style={"float": "right", "height": 90} # logo
    )
    link = html.A(logo, href="https://www.uma.es/#gsc.tab=0") # el enlace al que va el logo
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)])


# Load URLs and classes
RANDOM_URLS = open("urls/random_skin_cancer.txt").read().split("\n")[:-1] # urls de las imagenes

# Load model 
model = tf.keras.models.load_model('models/v2_model_06')


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]) # no se que hace
server = app.server

app.layout = dbc.Container(
    [
        Header("Clasificación y diagnóstico de cáncer de piel", app), # titulo que va a la funcion Header
        html.Hr(),
        dbc.Row( # botones de abajo en dos columnas
            [
                dbc.Col(
                    md=6,
                    children=[
                        dcc.Graph(id="stats-display"),
                        html.P("Introducir URL de la imagen:"), # introducir imagen
                        dbc.Input(
                            id="input-url",
                            placeholder='Insertar URL...',
                        ),
                    ],
                ),
                dbc.Col(
                    md=6,
                    children=[
                        dcc.Graph(id="image-display"),
                        html.P("Obtener clasificación:"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Ejecutar", id="btn-run", color="primary", n_clicks=0 # ejecutar
                                ),
                                dbc.Button(
                                    "Random", # boton para una imagen aleatoria que no me hara falta
                                    id="btn-random",
                                    color="primary",
                                    outline=True,
                                    n_clicks=0,
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback( # cuando pulso el boton random (input)
    [Output("btn-run", "n_clicks"), Output("input-url", "value")], # el output es rellenar con una url aleatoria y ejecutarlo
    [Input("btn-random", "n_clicks")], # el input es el boton random
    [State("btn-run", "n_clicks")], # no se que es state
)
def randomize(n_random, n_run): # lo que hace el boton random -- output
    return n_run + 1, RANDOM_URLS[n_random % len(RANDOM_URLS)] 


@app.callback(# cuando pulso el boton ejecutar
    [Output("image-display", "figure"), Output("stats-display", "figure")], # el output son los dos graficos
    [Input("btn-run", "n_clicks"), Input("input-url", "n_submit")], # tiene en cuenta el boton run y la url introducida
    [State("input-url", "value")], 
)
def run_model(n_clicks, n_submit, url): # lo que hace cuando se pulsa ejecutar -- output
    try:
        imagen= Image.open(requests.get(url, stream=True).raw) # carga la imagen
        #imagen = tf.keras.preprocessing.image.load_img(url, target_size=(224,224)) # como pongo la ruta
    except Exception as e:
        print(e)
        return px.scatter(title="Error: " + e)

    # ajustar la imagen para pasarsela al modelo
    imagen = imagen.resize((224,224))
    imagen_array = tf.keras.preprocessing.image.img_to_array(imagen)/255
    imagen = np.expand_dims(imagen_array, axis=0) # añadirla a una lista

    # hacer la prediccion
    prediction = model.predict(imagen)

    # establecer las clases
    class_2_indices = {'melanoma': 0, 'nevus': 1, 'seborrheic_keratoses': 2}
    indices_2_class = {v: k for k, v in class_2_indices.items()}

    # prediccion final
    image_idx = np.argmax(prediction[0]) # como indice
    prediction_string = indices_2_class[image_idx] # como clase

    # plot con la imagen y prediccion, explicabilidad
    imagen = inc_net.preprocess_input(imagen_array)
    explainer = li.LimeImageExplainer()
    explanation = explainer.explain_instance(imagen.astype('double'), 
                                         model.predict, 
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                            positive_only=False, 
                                            num_features=10, 
                                            hide_rest=False)
    temp_image = tf.keras.utils.array_to_img(temp)
                                                                               
    title = "Prediction: {}".format(prediction_string)
    fig = px.imshow(temp_image, title=title)   # plotea la imagen --> con la explicabilidad

    # plot con la probabilidad de prediccion
    scores_fig = px.bar( # plot de predicciones
        x=prediction[0], 
        y=['melanoma', 'nevus', 'seborrheic keratosis'],
        labels=dict(x="Probabilidad", y="Tipo de lesión"),
        title="Predicciones",
        orientation="h",
    )

    return fig, scores_fig


if __name__ == "__main__":
    app.run_server(debug=True)
