import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

app = Flask(__name__, static_folder='frontend')

# Global variables to store our ML artifacts in memory
DATA = None
PCA_MODEL = None
LR_MODEL = None
SCALER = None
PCA_RESULT = None

def load_and_train():
    global DATA, PCA_MODEL, LR_MODEL, SCALER, PCA_RESULT
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, 'dataset', 'traffic.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Dataset not found. Please place traffic.csv inside dataset/")
        return
        
    # Feature Engineering for Metro Interstate Dataset
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['temp_c'] = df['temp'] - 273.15  # Kelvin to Celsius
    
    df.fillna(df.mean(numeric_only=True), inplace=True)
    DATA = df

    # Features and Target
    X = df[['hour', 'temp_c', 'clouds_all', 'rain_1h']]
    y = df['traffic_volume']
    
    SCALER = StandardScaler()
    X_scaled = SCALER.fit_transform(X) # Standardize the features
    
    PCA_MODEL = PCA(n_components=2)
    pca_transformed = PCA_MODEL.fit_transform(X_scaled)
    
    # 1000 points sample so browser doesn't crash from 48000 points
    sample_indices = np.random.choice(len(df), size=min(1000, len(df)), replace=False)
    
    PCA_RESULT = {
        'pc1': pca_transformed[sample_indices, 0].tolist(),
        'pc2': pca_transformed[sample_indices, 1].tolist(),
        'variance_ratio': PCA_MODEL.explained_variance_ratio_.tolist()
    }
    
    # Train Linear Regressor
    LR_MODEL = LinearRegression()
    LR_MODEL.fit(X.values, y.values) 
    print("Models retrained successfully on new Metro dataset!")

# Call it upon startup
load_and_train()

@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

@app.route('/api/data', methods=['GET'])
def get_data():
    if DATA is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    # Sample 400 for visualization scatter plot
    sample = DATA.sample(n=min(400, len(DATA)), random_state=42).sort_values(by='hour')
    return jsonify(sample[['hour', 'traffic_volume', 'temp_c', 'clouds_all', 'rain_1h']].to_dict(orient='records'))

@app.route('/api/pca', methods=['GET'])
def get_pca():
    if PCA_RESULT is None:
        return jsonify({'error': 'PCA not computed'}), 500
    return jsonify({
        'scatter_data': [{'pc1': pc1, 'pc2': pc2} for pc1, pc2 in zip(PCA_RESULT['pc1'], PCA_RESULT['pc2'])],
        'variance_ratio': PCA_RESULT['variance_ratio']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if LR_MODEL is None:
        return jsonify({'error': 'Model not trained'}), 500
    data = request.json
    try:
        hour = float(data.get('hour', 12))
        temp_c = float(data.get('temperature', 20))
        clouds = float(data.get('clouds_all', 60))
        rain = float(data.get('rain_1h', 0))
        
        input_data = [[hour, temp_c, clouds, rain]]
        pred = LR_MODEL.predict(input_data)[0]
        pred = max(0, int(pred)) # No negative traffic
        
        # Calculate PCA for the predicted point using the new schema
        input_df = pd.DataFrame(input_data, columns=['hour', 'temp_c', 'clouds_all', 'rain_1h'])
        scaled_input = SCALER.transform(input_df)
        pca_coords = PCA_MODEL.transform(scaled_input)[0]

        return jsonify({
            'prediction': pred,
            'pca_pc1': float(pca_coords[0]),
            'pca_pc2': float(pca_coords[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights', methods=['GET'])
def get_insights():
    if DATA is None:
        return jsonify({'error': 'Data missing'}), 500
    
    peak_hour = DATA.groupby('hour')['traffic_volume'].mean().idxmax()
    return jsonify({
        'peak_hour': int(peak_hour),
        'avg_traffic': int(DATA['traffic_volume'].mean()),
        'total_records': len(DATA)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
