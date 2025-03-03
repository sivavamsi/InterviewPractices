from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)


def report(testing_data, anomalous_periods):
    valid_anomalies = anomalous_periods[(anomalous_periods > 2) & (anomalous_periods <= 14)]

    if valid_anomalies.index.to_list():
        valid_anomalous_time_ranges = []
        for group in valid_anomalies.index:
            anomaly_group_data = testing_data[testing_data['Anomaly_Group'] == group]
            start_date = anomaly_group_data.index.min()
            end_date = anomaly_group_data.index.max()
            valid_anomalous_time_ranges.append((start_date, end_date))
        anomalous_periods = pd.DataFrame(valid_anomalous_time_ranges, columns=['start', 'end'])
    else:
        anomalous_periods = pd.DataFrame()
    return anomalous_periods

@app.route('/predict', methods=['POST'])
def predict_anomalies():
    # Parse the input data
    input_data = request.get_json()
    print(input_data)


    # Convert input data to DataFrame
    test_data = pd.DataFrame(input_data['data'])

    # Apply scaling
    test_data_scaled = scaler.transform(test_data)

    # Make predictions
    predictions = model.predict(test_data_scaled)

    # Add results to a DataFrame
    test_data['Anomaly'] = predictions
    test_data['Date'] = pd.date_range(start=input_data['start_date'], periods=len(test_data))
    test_data.set_index("Date", inplace=True)

    # Group anomalies to identify periods
    test_data['Anomaly_Group'] = (test_data['Anomaly'] == -1).astype(int).diff().ne(0).cumsum()
    anomaly_groups = test_data[test_data['Anomaly'] == -1].groupby('Anomaly_Group').size()

    # Filter periods lasting >2 days and <=14 days
    #valid_anomalies = anomaly_groups[(anomaly_groups > 2) & (anomaly_groups <= 14)]
    details = report(test_data, anomaly_groups)
    # Prepare the output
    result = {
        "number_of_anomalous_periods": details.shape[0],
        "details": details.to_dict()
    }

    return jsonify(result)


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
