import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = "C:\\Users\\super\\Documents\\projects\\Yayscires"

# Load the confidence scores from CSV
def load_confidence_scores(file_path):
    """Load the confidence scores from a CSV file."""
    scores_df = pd.read_csv(file_path)
    return scores_df

# Perform T-test between two models
def t_test_between_models(scores_df, model_1, model_2, thresh=0.25):
    """Perform a T-test between the confidence scores of two models."""
    # Filter confidence scores by model_name
    model_1_scores = scores_df[(scores_df['modelname'] == model_1) & (scores_df['confidence'] >= thresh)]['confidence']
    model_2_scores = scores_df[(scores_df['modelname'] == model_2) & (scores_df['confidence'] >= thresh)]['confidence']

    print(model_1_scores)
    print(model_2_scores)
    
    t_stat, p_value = stats.ttest_ind(model_1_scores, model_2_scores)
    print(f"T-test between {model_1} and {model_2}:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # Interpret the p-value
    if p_value < 0.05:
        print(f"The difference between {model_1} and {model_2} is statistically significant.")
    else:
        print(f"No statistically significant difference between {model_1} and {model_2}.")

# Perform ANOVA between multiple models
def anova_between_models(scores_df, models):
    """Perform ANOVA between the confidence scores of multiple models."""
    model_scores = []
    
    for model in models:
        model_scores.append(scores_df[scores_df['modelname'] == model]['confidence'])
    
    f_stat, p_value = stats.f_oneway(*model_scores)
    print("ANOVA between models:")
    print(f"F-statistic: {f_stat}, P-value: {p_value}")

    # Interpret the p-value
    if p_value < 0.05:
        print(f"There is a statistically significant difference between the models.")
    else:
        print(f"No statistically significant difference between the models.")

# Visualize confidence scores distribution using a boxplot
def visualize_scores_boxplot(scores_df, models):
    """Visualize the distribution of confidence scores using a boxplot."""
    plt.figure(figsize=(10, 6))
    
    # Filter scores for the specified models and create a long-format DataFrame
    filtered_df = scores_df[scores_df['modelname'].isin(models)]
    
    # Plot the boxplot using seaborn
    sns.boxplot(data=filtered_df, x='modelname', y='confidence', palette='Set2')
    
    plt.title('Distribution of Confidence Scores by Model')
    plt.xlabel('Model')
    plt.ylabel('Confidence Score')
    plt.show()

# Visualize confidence scores comparison using a bar plot
def visualize_scores_barplot(scores_df, models):
    """Visualize the average confidence scores using a bar plot."""
    # Filter scores for the specified models and calculate mean confidence
    mean_scores = scores_df[scores_df['modelname'].isin(models)].groupby('modelname')['confidence'].mean().reset_index()
    
    # Plot the bar plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(mean_scores['modelname'], mean_scores['confidence'], color='skyblue')
    
    plt.title('Average Confidence Scores by Model')
    plt.xlabel('Model')
    plt.ylabel('Average Confidence Score')
    plt.show()

# Main function to run the tests and analysis
def run_tests_and_analysis():
    # File path where the confidence scores are saved
    file_path = os.path.join(ROOT, 'zcode', 'confidences.csv')  # Update with the actual path

    # Load the confidence scores data
    scores_df = load_confidence_scores(file_path)

    # List of models to compare
    models = ['Model_16_G', 'Model_16_NIR', 'Model_16_R', 'Model_16_RE', 'Model_16_RGB']

    t_test_between_models(scores_df, models[0], models[1], 0.25)
    #anova_between_models(scores_df, models)

    #visualize_scores_boxplot(scores_df, models)
    #visualize_scores_barplot(scores_df, models)

# Call the main function to run the analysis
run_tests_and_analysis()

