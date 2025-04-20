import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
import numpy as np

# --- Generic Plotting Functions ---

def plot_histogram(df, column, bins='auto', title=None, xlabel=None, ylabel=None, show=True):
    """
    Plot a histogram for a specified column in a DataFrame.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to plot.
        bins (int, str): Number of bins or binning strategy.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        show (bool): Whether to show the plot immediately.
    """
    data = df[column].dropna()
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title or f'Histogram of {column}')
    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel or 'Count')
    plt.grid(True)
    if show:
        plt.show()

def plot_upset_generic(df, indicator_columns, max_combinations=20, title=None, show=True):
    """
    Plot an UpSet plot showing the most common combinations of binary indicator columns.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        indicator_columns (list): List of columns to use as indicators (should be boolean or 0/1).
        max_combinations (int): Max number of combinations to display.
        title (str): Plot title.
        show (bool): Whether to show the plot immediately.
    """
    upset_data = from_indicators(indicator_columns, data=df)
    upset = UpSet(upset_data, subset_size='count', show_counts=True)
    upset.plot()
    plt.title(title or 'UpSet Plot')
    if show:
        plt.show()

# --- End of myplot.py ---

# To use these functions, uncomment them and call with your DataFrame and columns of interest.
