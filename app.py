from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            rm = float(request.form['rm'])
            lstat = float(request.form['lstat'])
            ptratio = float(request.form['ptratio'])
        except ValueError:
            return render_template('index.html', error='Invalid input. Please enter numeric values.')

        new_data = np.array([[rm, lstat, ptratio]])
        new_data_scaled = scaler.transform(new_data)
        predicted_price = model.predict(new_data_scaled)[0]
        return render_template('index.html', prediction=f'Predicted Price: {predicted_price:.2f}')

if __name__ == '__main__':
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    target = pd.DataFrame(boston.target, columns=["MEDV"])
    df = pd.concat([data, target], axis=1)

    selected_features = ["RM", "LSTAT", "PTRATIO"]
    X = df[selected_features]
    y = df["MEDV"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    app.run(debug=True)
