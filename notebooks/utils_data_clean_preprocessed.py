import numpy as np
import pandas as pd

# Columns to drop at final stage
columns_to_drop = [
    'rider_id',
    'restaurant_latitude',
    'restaurant_longitude',
    'delivery_latitude',
    'delivery_longitude',
    'order_date',
    "order_time_hour",
    "order_day",
    "city_name",
    "order_day_of_week",
    "order_month"
]


# ---------------------------------------------------
# Time of day function
# ---------------------------------------------------
def time_of_day(ser):
    return pd.cut(
        ser,
        bins=[0, 6, 12, 17, 20, 24],
        right=True,
        labels=["after_midnight", "morning", "afternoon", "evening", "night"]
    )


# ---------------------------------------------------
# Change column names
# ---------------------------------------------------
def change_column_names(data: pd.DataFrame):

    return (
        data.rename(str.lower, axis=1)
        .rename({
            "delivery_person_id": "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"
        }, axis=1)
    )


# ---------------------------------------------------
# Data cleaning and feature engineering
# ---------------------------------------------------
def data_cleaning(data: pd.DataFrame):

    minors_data = data.loc[data['age'].astype(float) < 18]
    minor_index = minors_data.index.tolist()

    six_star_data = data.loc[data['ratings'] == "6"]
    six_star_index = six_star_data.index.tolist()

    return (
        data
        .drop(columns="id", errors="ignore")
        .drop(index=minor_index)
        .drop(index=six_star_index)
        .replace("NaN ", np.nan)

        .assign(

            # city extraction
            city_name=lambda x: x['rider_id'].str.split("RES").str.get(0),

            # numeric conversion
            age=lambda x: x['age'].astype(float),
            ratings=lambda x: x['ratings'].astype(float),

            # absolute lat long
            restaurant_latitude=lambda x: x['restaurant_latitude'].abs(),
            restaurant_longitude=lambda x: x['restaurant_longitude'].abs(),
            delivery_latitude=lambda x: x['delivery_latitude'].abs(),
            delivery_longitude=lambda x: x['delivery_longitude'].abs(),

            # datetime conversion
            order_date=lambda x: pd.to_datetime(x['order_date'], dayfirst=True),

            order_day=lambda x: x['order_date'].dt.day,
            order_month=lambda x: x['order_date'].dt.month,

            order_day_of_week=lambda x: (
                x['order_date']
                .dt.day_name()
                .str.lower()
            ),

            is_weekend=lambda x: (
                x['order_date']
                .dt.day_name()
                .isin(["Saturday", "Sunday"])
                .astype(int)
            ),

            # order time
            order_time=lambda x: pd.to_datetime(
                x['order_time'],
                format='mixed'
            ),

            order_picked_time=lambda x: pd.to_datetime(
                x['order_picked_time'],
                format='mixed'
            ),

            # pickup time
            pickup_time_minutes=lambda x: (
                (x['order_picked_time'] - x['order_time'])
                .dt.seconds / 60
            ),

            # hour feature
            order_time_hour=lambda x: x['order_time'].dt.hour,

            # time of day
            order_time_of_day=lambda x: (
                x['order_time_hour'].pipe(time_of_day)
            ),

            # categorical cleaning
            weather=lambda x: (
                x['weather']
                .str.replace("conditions ", "", regex=False)
                .str.lower()
                .replace("nan", np.nan)
            ),

            traffic=lambda x: x["traffic"].str.rstrip().str.lower(),

            type_of_order=lambda x: (
                x['type_of_order']
                .str.rstrip()
                .str.lower()
            ),

            type_of_vehicle=lambda x: (
                x['type_of_vehicle']
                .str.rstrip()
                .str.lower()
            ),

            festival=lambda x: x['festival'].str.rstrip().str.lower(),

            city_type=lambda x: x['city_type'].str.rstrip().str.lower(),

            multiple_deliveries=lambda x: (
                x['multiple_deliveries'].astype(float)
            ),

            # target column cleaning
            time_taken=lambda x: (
                x['time_taken']
                .str.replace("(min) ", "", regex=False)
                .astype(int)
            )
        )

        .drop(columns=["order_time", "order_picked_time"])
    )


# ---------------------------------------------------
# Clean lat long
# ---------------------------------------------------
def clean_lat_long(data: pd.DataFrame, threshold=1):

    location_columns = [
        'restaurant_latitude',
        'restaurant_longitude',
        'delivery_latitude',
        'delivery_longitude'
    ]

    return (
        data
        .assign(**{
            col: lambda x: np.where(
                x[col] < threshold,
                np.nan,
                x[col]
            )
            for col in location_columns
        })
    )


# ---------------------------------------------------
# Haversine distance
# ---------------------------------------------------
def calculate_haversine_distance(df):

    lat1 = df['restaurant_latitude']
    lon1 = df['restaurant_longitude']
    lat2 = df['delivery_latitude']
    lon2 = df['delivery_longitude']

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    distance = 6371 * c

    return df.assign(distance=distance)


# ---------------------------------------------------
# Create distance category
# ---------------------------------------------------
def create_distance_type(data: pd.DataFrame):

    return (
        data
        .assign(
            distance_type=pd.cut(
                data["distance"],
                bins=[0, 5, 10, 15, 25],
                right=False,
                labels=[
                    "short",
                    "medium",
                    "long",
                    "very_long"
                ]
            )
        )
    )


# ---------------------------------------------------
# Drop columns
# ---------------------------------------------------
def drop_columns(data: pd.DataFrame, columns: list):

    return data.drop(columns=columns, errors="ignore")


# ---------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------
def perform_data_cleaning(data: pd.DataFrame):

    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns, columns=columns_to_drop)
    )

    return cleaned_data.dropna()


# ---------------------------------------------------
# Main execution
# ---------------------------------------------------
if __name__ == "__main__":

    DATA_PATH = "swiggy.csv"

    df = pd.read_csv(DATA_PATH)

    print("Swiggy data loaded successfully")

    cleaned_df = perform_data_cleaning(df)

    print("Cleaned data shape:", cleaned_df.shape)

    print(cleaned_df.head())