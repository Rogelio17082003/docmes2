from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el scaler ajustado
model = joblib.load('modelo.joblib')
scaler = joblib.load('scaler.joblib')# esto sirve para escalar  los datos de las cajas de texto ver proyecto de fin de mes 1
app.logger.debug('Modelo y scaler cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
           # Obtener los datos enviados en el request
       
        spec_score = request.form.get('spec_score')
        ram = request.form.get('ram')
        external_memory = request.form.get('external_memory')
        company = request.form.get('company')
        screen_resolution = request.form.get('screen_resolution')
        processor = request.form.get('Processor')
        


        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[spec_score, ram,external_memory,company,screen_resolution,processor]], columns=['Spec_score','Ram', 'External_Memory', 'company', 'Screen_resolution', 'Processor'])
        app.logger.debug(f'DataFrame creado: {data_df}')
       



        
        # Escalar los datos utilizando el scaler ajustado anteriormente,se deben de escalar los datos mandados ya que en el standar scales
        # con el que hicimos el entrenamiendo del arbol de desicion esta entrenado con numeros escalados
        #scalerX = scaler.transform(data_df)
        #scaler_df = pd.DataFrame(scalerX, columns=data_df.columns)
        #app.logger.debug(f'DataFrame escalado: {scaler_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
       
        

        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=port)
