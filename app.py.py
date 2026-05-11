import streamlit as st
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

st.title("Seismic Data Query Tool")

@st.cache_resource
def load_data():
    with rasterio.open('IL_grid_predict_krigged_cropped.tif') as src:
        data = src.read(1).astype('float32')
        data[data == src.nodata] = np.nan
        
        # Using our "Geographic Truth" fix
        l, b, r, t = src.bounds
        left, right = min(l, r), max(l, r)
        bottom, top = min(b, t), max(b, t)
        
        x_coords = np.linspace(left, right, src.width)
        y_coords = np.linspace(bottom, top, src.height)
        
        interp = RegularGridInterpolator((y_coords, x_coords), data[::-1, :], 
                                         method='linear', bounds_error=False, fill_value=np.nan)
        return data[::-1, :], [left, right, bottom, top], interp

flipped_data, extent, interp_func = load_data()

# Sidebar for inputs
st.sidebar.header("Input Coordinates (ITM)")
x_input = st.sidebar.number_input("Easting (X)", value=extent[0] + 500.0)
y_input = st.sidebar.number_input("Northing (Y)", value=extent[2] + 500.0)

if st.sidebar.button("Query Value"):
    val = interp_func([y_input, x_input])[0]
    
    if not np.isnan(val):
        st.success(f"Value at ({x_input}, {y_input}): **{val:.4f}**")
    else:
        st.error("Coordinates are outside map bounds.")

    # Plotting
    fig, ax = plt.subplots()
    img = ax.imshow(flipped_data, extent=extent, origin='upper', cmap='terrain')
    plt.colorbar(img, ax=ax)
    ax.plot(x_input, y_input, 'ro', markersize=10, markeredgecolor='white')
    ax.set_aspect('equal')
    st.pyplot(fig)