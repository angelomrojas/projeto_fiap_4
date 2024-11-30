import uvicorn
from http import HTTPStatus
from fastapi import FastAPI
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()
model = keras.models.load_model('modelo_lstm.h5')
scaler = MinMaxScaler(feature_range=(0, 1))


@app.post("/predict/")
async def predict(data: list[float]):
    # Preprocess the input data (similar to your training data)
    look_back = 40  # Match your training look_back
    data = np.array(data).reshape(-1, 1)  # Reshape to 2D
    data = scaler.fit_transform(data)  # Scale the data

    predictions = np.array([])

    for i in range(3):
        input = data[-look_back:]
        input = input.reshape(-1, look_back, 1)
        # Make predictions using the loaded model
        prediction = model.predict(input)

        # Inverse transform the predictions to get actual values
        prediction = scaler.inverse_transform(prediction)
        data = np.append(data, prediction[0])
        predictions = np.append(predictions, prediction[0])
        
    print(predictions.tolist())
    return {"predictions": predictions.tolist()}
    

@app.get("/health", status_code=200)
async def root():
    return HTTPStatus.OK


if __name__ == '__main__':
    uvicorn.run("main:app", reload=False, host="0.0.0.0", port=8000)