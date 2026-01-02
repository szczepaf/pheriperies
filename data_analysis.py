"""A module for analyzing election and city connection data. Most importantly, it combines the datasets for election and city connections."""
import json
from pathlib import Path
import db.static as static

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from ydata_profiling import ProfileReport
import numpy as np


def population_histogram(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Plot a histogram of city populations using seaborn and matplotlib.

    Parameters
    ----------
    csv_filename : str
        Name of the CSV file with the combined data (must contain a 'population' column).
        By default, assumes 'db/combined_data.csv' in the same directory as this script.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Load data
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Create histogram
    # let the y axis be logarithmic to better see the distribution
    sns.histplot(df["population"], bins=30)
    plt.yscale("log")

    plt.xlabel("Population of city")
    plt.ylabel("Number of cities")
    plt.title("Distribution of city populations")
    plt.tight_layout()
    plt.show()


def election_percentage_histogram(csv_filename: str = "db/combined_data.csv") -> None:
    """
    Plot a histogram of election percentages using seaborn and matplotlib.

    Parameters
    ----------
    csv_filename : str
        Name of the CSV file with the combined data (must contain a 'population' column).
        By default, assumes 'db/combined_data.csv' in the same directory as this script.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Load data
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Create histogram
    sns.histplot(df["UCAST_PROC"], bins=100, kde=True)

    plt.xlabel("Election Percentage")
    plt.ylabel("Number of Cities")
    plt.title("Distribution of Election Percentages")
    plt.tight_layout()
    plt.show()


def distance_to_regional_city_histogram(
    csv_filename: str = "db/combined_data.csv",
) -> None:
    """
    Plot a histogram of distances to regional cities using seaborn and matplotlib.

    Parameters
    ----------
    csv_filename : str
        Name of the CSV file with the combined data (must contain a 'distance' column).
        By default, assumes 'db/combined_data.csv' in the same directory as this script.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Load data
    df = pd.read_csv(csv_path, encoding="utf-8")
    # min of these cols: Praha,Brno,Ostrava,Plzeň,Liberec,Olomouc,Ústí nad Labem,Hradec Králové,Pardubice,Zlín,České Budějovice,Jihlava,Karlovy Vary
    df["distance"] = df[static.REGIONAL_CITIES_NAMES].min(axis=1)
    # remove entries with distance == 0
    df = df[df["distance"] > 0]

    # Create histogram
    sns.histplot(df["distance"], bins=30, kde=True)
    plt.yscale("log")

    plt.xlabel("Distance to regional city (minutes)")
    plt.ylabel("Number of cities")
    plt.title("Distribution of distances to regional cities")
    plt.tight_layout()
    plt.show()



def analyze_party25_6_vs_distance(csv_filename: str = "db/combined_data.csv") -> None:
    """
    For each city, compute the summed percentage votes of parties 25 and 6
    and plot it against the city's distance from the regional city.
    Then evaluate linear dependence using simple linear regression.

    Parameters
    ----------
    csv_filename : str
        Name of the CSV file with the combined data. Must contain:
        - P25_PROC, P6_PROC (percent vote columns)
        - distance (distance to regional city)
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / csv_filename

    # Load data
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[(df["city_name"] != "Jesenice") & (df["city_name"] != "Orlová")]


    analysis_df = df.copy()
    # Prepare data: sum of percentages for parties 25 and 6
    analysis_df["sum_P25_P6_PROC"] = analysis_df["P25_PROC"] + analysis_df["P6_PROC"]

    analysis_df["distance"] = analysis_df[static.REGIONAL_CITIES_NAMES].min(axis=1)
    x = analysis_df["distance"]
    y = analysis_df["sum_P25_P6_PROC"]

    # Linear regression
    reg_result = linregress(x, y)

    # Plot scatter with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x, y=y, ci=95)

    plt.xlabel("Distance to regional city")
    plt.ylabel("Sum of vote share P25 + P6 (%)")
    plt.title("Sum of parties 25 and 6 vs distance to regional city")
    plt.tight_layout()
    plt.show()

    # Print regression evaluation
    print("Linear regression: sum(P25 + P6) vs distance")
    print(f"  slope:      {reg_result.slope:.4f} percentage points per distance unit")
    print(f"  intercept:  {reg_result.intercept:.4f}")
    print(f"  r-value:    {reg_result.rvalue:.4f}")
    print(f"  r-squared:  {reg_result.rvalue**2:.4f}")
    print(f"  p-value:    {reg_result.pvalue:.4e}")
    print(f"  std err:    {reg_result.stderr:.4f}")

    alpha = 0.05
    if reg_result.pvalue < alpha:
        print(
            f"At alpha = {alpha}, there is statistically significant linear dependence."
        )
    else:
        print(
            f"At alpha = {alpha}, there is no statistically significant linear dependence."
        )


def create_dataset() -> None:
    """
    Combine election_data.csv with verbose city connection data
    (city_connections_verbose.json) into combined_data.csv.

    Resulting CSV:
      - all original columns from election_data.csv
      - plus:
          population                     number of inhabitants
          one column per regional city   distance (minutes) to that regional city
      - the old 'regional_city' and 'distance' columns are removed (if present)
    """
    base_dir = Path(__file__).resolve().parent

    elections_path = base_dir / "db/election_data.csv"
    connections_path = base_dir / "db/city_connections_verbose.json"
    output_path = base_dir / "db/combined_data.csv"


    # Load election data
    elections_df = pd.read_csv(elections_path, encoding="utf-8")

    # Load verbose connection data: {"data": [ { "name", "population", "connections": {...} }, ... ]}
    with connections_path.open("r", encoding="utf-8") as f:
        connections_data = json.load(f)

    connections_df = pd.DataFrame(connections_data["data"])

    # Expand the 'connections' dict column into separate columns
    # (one for each regional city, in minutes)
    connections_expanded = connections_df["connections"].apply(pd.Series)

    # Keep only the regional cities we care about (and in a stable order)
    connections_expanded = connections_expanded[static.REGIONAL_CITIES_NAMES]

    # Build a flat connections table: city_name, population, and one column per regional city
    connections_flat = pd.concat(
        [connections_df[["name", "population"]], connections_expanded], axis=1
    ).rename(columns={"name": "city_name"})

    # Merge datasets on city_name; inner keeps only cities present in both datasets
    combined_df = pd.merge(
        elections_df,
        connections_flat,
        on="city_name",
        how="inner",
    )

    # Drop any legacy columns if they exist
    combined_df = combined_df.drop(columns=["regional_city", "distance"], errors="ignore")

    # Save result
    combined_df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    # create_dataset()
    # population_histogram()
    # election_percentage_histogram()
    # distance_to_regional_city_histogram()
    # analyze_party25_6_vs_distance()
    
    # read the created dataset
    df = pd.read_csv("db/combined_data.csv", encoding="utf-8")
    profile = ProfileReport(df, title="YData Profiling Report")
    print("Generating profiling report...")
    profile.to_file("your_report.html")


    
