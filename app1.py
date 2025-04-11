import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend non interactif pour éviter les erreurs GUI
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 Mo max
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'json', 'xml'}

# Vérifie si l'extension est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Chargement avec pandas uniquement
def load_file_with_pandas(file, ext):
    if ext == 'csv':
        return pd.read_csv(file)
    elif ext == 'json':
        return pd.read_json(file)
    elif ext == 'xml':
        content = file.read()
        # Convertir le XML en JSON (ou dictionnaire Python)
        df = pd.read_xml(io.BytesIO(content))
        
        # Si le XML contient des données imbriquées (comme l'adresse), il faut les aplatir
        # Utilisation de json_normalize pour aplatir les données imbriquées
        if 'adresse' in df.columns:
            df = pd.json_normalize(df.to_dict(orient='records'))
        
        return df
    else:
        raise ValueError("Format non pris en charge")

# Nettoyage de données
def clean_data(df):
    try:
        # Gestion des valeurs manquantes
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("INCONNU")

        # Suppression des doublons
        df.drop_duplicates(inplace=True)

        # Traitement des outliers
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df[col] = np.clip(df[col], lower, upper)

        return df
    except Exception as e:
        raise ValueError(f"Erreur lors du nettoyage des données: {e}")

# Création du boxplot
def create_boxplot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.select_dtypes(include=[np.number]).boxplot(ax=ax)
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

# Route principale pour upload et nettoyage
@app.route('/api/clean', methods=['POST'])
def upload_and_clean():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400

    file = request.files['file']
    filename = file.filename

    if filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    if not allowed_file(filename):
        return jsonify({"error": "Seuls les fichiers CSV, JSON ou XML sont acceptés"}), 400

    ext = filename.rsplit('.', 1)[1].lower()

    try:
        # Chargement du fichier
        df = load_file_with_pandas(file, ext)

        # Nettoyage
        cleaned_df = clean_data(df)

        # Création du boxplot
        boxplot_image_base64 = create_boxplot(cleaned_df)

        # Conversion du DataFrame nettoyé en CSV (en base64)
        output = io.BytesIO()
        cleaned_df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)

        return jsonify({
            "csv": base64.b64encode(output.getvalue()).decode('utf-8'),
            "boxplot_image": boxplot_image_base64
        })

    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement: {str(e)}"}), 500
# Route pour afficher le front-end
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True, port=5000)
