from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi.responses import JSONResponse



# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and preprocess dataset
file_path = 'Property Sales of Melbourne City.csv'
data = pd.read_csv(file_path)

# Define relevant features
key_features = ['Suburb', 'Rooms', 'Type', 'Price', 'Distance', 'Bathroom', 'Car', 'Landsize', 
                'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Regionname']
cleaned_data = data[key_features].ffill()

# Set up encoder and scaler, then save for consistent transformations
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(cleaned_data[['Suburb', 'Type', 'Regionname']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Suburb', 'Type', 'Regionname']))

scaler = MinMaxScaler()
numerical_columns = ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 
                     'YearBuilt', 'Lattitude', 'Longtitude']
cleaned_data[numerical_columns] = scaler.fit_transform(cleaned_data[numerical_columns])

# Combine transformed features
cleaned_data = pd.concat([cleaned_data.drop(columns=['Suburb', 'Type', 'Regionname']), encoded_df], axis=1)

# Split and train model
X = cleaned_data.drop(columns=['Price'])
y = cleaned_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
joblib.dump(forest_model, 'forest_model.pkl')

joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load trained model, encoder, and scaler
forest_model = joblib.load('forest_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
gb_model = joblib.load('gb_model.pkl')

# Define the input structure for prediction
class PredictionRequest(BaseModel):
    Suburb: str
    Rooms: int
    Type: str
    Bathroom: int
    Car: int
    BuildingArea: float = None
    Landsize: float = None
    Distance: float = None
    model_type: str  # 'forest' or 'gb'


# Function to fill missing features with default values
def fill_missing_features(data):
    defaults = {
        "Landsize": 500.0,
        "BuildingArea": 150.0,
        "YearBuilt": 2000,
        "Car": 1,
        "Lattitude": -37.8136,
        "Longtitude": 144.9631,
        "Distance": 5.0,
        "Regionname": "Northern Metropolitan"  # Use a common value from the dataset
    }
    for key, value in defaults.items():
        if key not in data or data[key] is None:
            data[key] = value
    return data

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert request data to dictionary and fill missing values
        data = request.dict()
        data = fill_missing_features(data)
        model_type = data.pop("model_type")  # Extract model type

        # Prepare DataFrame with one row of input data
        input_df = pd.DataFrame([data])

        # Encode categorical features using the pre-trained encoder
        encoded_input = encoder.transform(input_df[['Suburb', 'Type', 'Regionname']])
        encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['Suburb', 'Type', 'Regionname']))

        # Concatenate numerical and encoded categorical features
        input_df = pd.concat([input_df.drop(columns=['Suburb', 'Type', 'Regionname']), encoded_input_df], axis=1)

        # Scale numerical columns to match training data
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

        # Ensure input data matches model input structure (add missing columns with 0s)
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]

        # Select model based on `model_type`
        model = forest_model if model_type == "forest" else gb_model
        prediction = model.predict(input_df.to_numpy())[0]

        return {"prediction": prediction}

    except Exception as e:
        print(f"Error during prediction: {e}")  # Print full error message
        return {"error": "An error occurred during prediction"}


# Endpoint to get available suburbs
@app.get("/suburbs")
async def get_suburbs():
    unique_suburbs = data['Suburb'].dropna().unique().tolist()
    return {"suburbs": unique_suburbs}

# Endpoint to get options for a specific suburb
@app.get("/options/{suburb}")
async def get_options(suburb: str):
    suburb_data = data[data['Suburb'] == suburb]
    rooms = sorted(suburb_data['Rooms'].dropna().unique().tolist())
    types = sorted(suburb_data['Type'].dropna().unique().tolist())
    bathrooms = sorted(suburb_data['Bathroom'].dropna().unique().astype(int).tolist())
    cars = sorted(suburb_data['Car'].dropna().unique().astype(int).tolist())
    regions = sorted(suburb_data['Regionname'].dropna().unique().tolist())  # Added Regionname options

    return {
        "rooms": rooms,
        "types": types,
        "bathrooms": bathrooms,
        "cars": cars,
        "regionname": regions  # Add regionname to response
    }

@app.get("/price-distribution/{suburb}")
async def get_price_distribution(suburb: str, bins: int = Query(10, ge=1, le=50)):
    suburb_data = data[data['Suburb'] == suburb]

    if suburb_data.empty:
        return {"error": f"No data available for suburb: {suburb}"}

    # Calculate histogram data
    price_counts, bin_edges = np.histogram(suburb_data['Price'].dropna(), bins=bins)

    # Convert numpy data to native Python types
    price_distribution = {
        "bin_edges": bin_edges.tolist(),  # Convert numpy array to list
        "price_counts": price_counts.tolist(),  # Convert numpy array to list
        "average_price": float(suburb_data['Price'].mean()),  # Convert numpy float to Python float
        "min_price": float(suburb_data['Price'].min()),  # Convert numpy int to Python int
        "max_price": float(suburb_data['Price'].max()),  # Convert numpy int to Python int
    }

    return price_distribution

# Actual vs. Predicted Endpoint (Filtered by Suburb Only)
@app.get("/actual-vs-predicted/{suburb}")
async def actual_vs_predicted(suburb: str):
    try:
        # Filter the dataset by the selected suburb
        suburb_data = data[data['Suburb'] == suburb]
        
        # If not enough data points match, return an error
        if suburb_data.empty:
            return JSONResponse(content={"error": f"No data available for suburb: {suburb}"}, status_code=400)

        # Select a batch of data from the filtered subset for testing
        test_data = suburb_data.sample(min(len(suburb_data), 20), random_state=42)
        actual_prices = test_data['Price'].tolist()

        # Prepare test data for prediction
        test_data = test_data.drop(columns=['Price'])
        encoded_input = encoder.transform(test_data[['Suburb', 'Type', 'Regionname']])
        encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['Suburb', 'Type', 'Regionname']))
        test_data = pd.concat([test_data.drop(columns=['Suburb', 'Type', 'Regionname']), encoded_input_df], axis=1)
        test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

        # Ensure input data matches model input structure (add missing columns with 0s)
        for col in X.columns:
            if col not in test_data.columns:
                test_data[col] = 0
        test_data = test_data[X.columns]

        # Predict prices for the test batch
        predicted_prices = forest_model.predict(test_data).tolist()

        return JSONResponse(content={"actual": actual_prices, "predicted": predicted_prices})

    except Exception as e:
        print("Error fetching actual vs predicted data:", e)
        return JSONResponse(content={"error": "Failed to load actual vs predicted data."}, status_code=500)


@app.get("/average-price-by-year/{suburb}")
async def average_price_by_year(suburb: str):
    try:
        # Filter data for the selected suburb
        suburb_data = data[data['Suburb'] == suburb]
        
        if suburb_data.empty:
            return {"error": f"No data available for suburb: {suburb}"}

        # Group by YearBuilt and calculate average price
        average_price_by_year = (
            suburb_data.groupby('YearBuilt')['Price']
            .mean()
            .dropna()
            .reset_index()
            .sort_values('YearBuilt')
        )

        # Convert the result to a JSON serializable format
        return {
            "year_built": average_price_by_year['YearBuilt'].tolist(),
            "average_price": average_price_by_year['Price'].tolist()
        }
    except Exception as e:
        print("Error fetching average price by year:", e)
        return {"error": "Failed to load data."}