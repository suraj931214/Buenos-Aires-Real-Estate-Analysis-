import streamlit as st
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

#from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
 
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline


###########
st.title("Buenos Aires Aparments")
st.header("Welcome to my real estate website")

st.markdown("""
    Here you can get to know about apartments price variation.
    Available in buenos aries with exact location, neighbourhood and area.
""")

st.subheader("lookout first sight of your apartment ")
########

files=glob("data\\buenos-aires-real-*")
 

def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)
    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]
    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]
    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)
    # Get place name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)
    # Drop column which has more then 50% null values
    df.drop(columns=['floor','expenses'],inplace=True)
    #dropping low and high cardinality features
    df.drop(columns=['operation','property_type','currency','properati_url'],inplace=True)
    # dro leaky (same type of columns ) columns
    df.drop(columns=[
        'price',
        'price_aprox_local_currency',
        'price_per_m2',
        'price_usd_per_m2'
    ],
           inplace=True)
    #drop columns with multi collinarity (high collinearity between features col
    df.drop(columns=[
        'surface_total_in_m2',
        'rooms'
    ],
           inplace=True)
  
 
    return df

frames=[wrangle(file) for file in files]
df=pd.concat(frames,ignore_index=True)
# print(df.head())

fig=px.scatter_mapbox(
    df,
    lat='lat',
    lon='lon',
    hover_name="neighborhood",
    hover_data=["surface_covered_in_m2"],
    zoom=10,
    height=600
)

fig.update_layout(mapbox_style="open-street-map")

######
st.plotly_chart(fig,use_container_width=True)
st.text("this map will guide you throughout your apartment location with price and area")
######

 
#st.dataframe(df.head(7))

#Dividing our data into train and test
target='price_aprox_usd'
features=['surface_covered_in_m2','lat','lon','neighborhood']

X_train=df[features]
y_train=df[target]
# X_train.shape,y_train.shape


# Build a baseline model
y_mean=y_train.mean()
y_pred_baseline=[y_mean]*len(y_train)

# print('Mean appartment Price',y_mean)
# print('Baseline MAE',mean_absolute_error(y_train,y_pred_baseline))

model=make_pipeline(
    OneHotEncoder(handle_unknown='ignore'),
    SimpleImputer(),
    Ridge()
    
)
model.fit(X_train,y_train)


# y_pred_training=model.predict(X_train)
# print("Training MAE:", mean_absolute_error(y_train,y_pred_training))


# X_test = pd.read_csv("data/buenos-aires-test-features.csv")
# y_pred_test = pd.Series(model.predict(X_test))
# y_pred_test.head()

########
st.title('Price Predictor')
st.subheader('Just enter your requirement i will tell you price you need to invest in your apartment 🏠')
st.text('Take help of above map')

# neighborhood=st.text_input("Enter Neighborhood:")
# area=st.number_input("Enter Area (in meter square)",min_value=0)
# lat=st.number_input("Enter Latitude")
# lon=st.number_input("Enter Longitude")

# sub_btn=st.button("Predict Price")
########


def make_prediction(area, lat, lon, neighborhood):
    data={
        'surface_covered_in_m2':area,
        'lat':lat,
        'lon':lon,
        'neighborhood':neighborhood
    }
    df_values=pd.DataFrame(data,index=[0])
    prediction = model.predict(df_values).round(2)[0]
    return prediction
#######
area=st.slider(
    "Select Area (in meter square)",
    min_value=int(X_train['surface_covered_in_m2'].min()),
    max_value=int(X_train['surface_covered_in_m2'].max()),
    value=int(X_train['surface_covered_in_m2'].mean())
)
area=st.number_input(
    "or Enter Area (in meter square)",
    min_value=int(X_train['surface_covered_in_m2'].min()),
    max_value=int(X_train['surface_covered_in_m2'].max()),
    value=area
)
lat=st.slider(
    "Select Latitude",
    min_value=float(X_train['lat'].min()),
    max_value=float(X_train['lat'].max()),
    step=0.01,
    value=float(X_train['lat'].mean())
)
lat=st.number_input(
    "Select Latitude",
    min_value=float(X_train['lat'].min()),
    max_value=float(X_train['lat'].max()),
    value=lat
)
lon=st.slider(
    "Select Latitude",
    min_value=float(X_train['lon'].min()),
    max_value=float(X_train['lon'].max()),
    step=0.01,
    value=float(X_train['lon'].mean())
)
lon=st.number_input(
    "Select Latitude",
    min_value=float(X_train['lon'].min()),
    max_value=float(X_train['lon'].max()),
    value=lon
)
neighborhood=st.selectbox(
    'Select Neighborhood',
    options=sorted(X_train['neighborhood'].unique())
)

#######
if st.button("Predict Price"):
    price=make_prediction(area,lat,lon,neighborhood)
    st.success(f"💰 Estimated Price: ${price}")
###########
#make_prediction(110,-34.60,-58.46,"villa crespo")



