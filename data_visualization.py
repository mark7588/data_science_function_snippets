# Display scatter plot between two variables x and y 
def lib_for_data_visulization():
    import matplotlib.pyplot as plt  # For scatter plots and visualization
    import numpy as np               # For numerical calculations, e.g., polyfit
    import pandas as pd              # For DataFrame operations (if you're working with tabular data)


def display_scatter_plot(x, y, x_label, y_label):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Scatter plot of {x_label} vs {y_label}")
    plt.show()

# Display scatter plot with trend line between two variables x and y 
def display_scatter_plot_trend_line(x, y, x_label, y_label):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    slope, intercept = np.polyfit(x, y, 1)
    plt.plot(x, slope * x + intercept, color='red', label='Trend Line')

    plt.title(f"Scatter plot of {x_label} vs {y_label}")
    plt.show()

# Extract quantitative field from all of the fields in the table 
def extract_numeric_data (df):
    categorical_fields = [col for col in df.columns if df[col].dtype == 'object']
    numeric_fields = [col for col in df.columns if df[col].dtype != 'object']
    return numric_fields 


