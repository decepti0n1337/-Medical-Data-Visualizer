import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column

# Create DataFrame

# Calculate BMI
df['height_m'] = df['height'] / 100  # Convert height to meters
df['bmi'] = df['weight'] / (df['height_m'] ** 2)  # Calculate BMI

# Define function to determine overweight status
def is_overweight(bmi):
    if bmi > 25:
        return 1
    else:
        return 0

# Apply function to create overweight column
df['overweight'] = df['bmi'].apply(is_overweight)

# Drop intermediate columns
df.drop(['height_m', 'bmi'], axis=1, inplace=True)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)



# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    selected_columns = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight', 'cardio']
    df_categorical = df[selected_columns]

    melted_df = pd.melt(df_categorical, id_vars=['cardio'], var_name='variable', value_name='value')

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    grouped_df = melted_df.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Draw the catplot with 'sns.catplot()'
    plot = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=grouped_df, height=5, aspect=1.2)

    # Get the figure for the output
    fig = plot.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975))]


    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=0.5)
    fig = plt.gcf()

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
