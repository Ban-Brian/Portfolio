import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("Data Information:")
    print(df.info())
    print("\n")

    print("Descriptive Statistics:")
    print(df.describe())
    print("\n")

    print("Missing Values:")
    print(df.isnull().sum())
    print("\n")

    print("Correlation Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    print("Histograms for Numeric Columns:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Histograms of Numeric Columns")
    plt.show()

    print("Boxplots for Numeric Columns:")
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=col)
        plt.title(f"Boxplot for {col}")
        plt.show()

    print("Bar Plots for Categorical Columns:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=col)
        plt.title(f"Bar Plot for {col}")
        plt.show()
