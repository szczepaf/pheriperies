import sys

import pandas as pd
from city import City
from connector import Connector
import json

sys.stdout.reconfigure(encoding="utf-8")


def create_cities_dict():
    """Return a dict of cities with population at least 10,000"""
    df = pd.read_csv("population_data.csv", index_col=False)

    # Convert population to numeric; invalid / empty values become NaN
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    # Filter municipalities with population at least 10,000
    df_10k = df[df["population"] >= 10_000].copy()
    # create a dict where cities are queryable by name
    cities_dict = {}

    for _, row in df_10k.iterrows():
        name = str(row["municipality_name"])
        population = int(row["population"])

        city = City(name, population)
        # regional_city and distance already defaulted in __init__
        cities_dict[name] = city

    return cities_dict

def main():
    # load cities dict
    cities_dict = create_cities_dict()
    regional_cities_names = [
        "Praha",
        "Brno",
        "Ostrava",
        "Plzeň",
        "Liberec",
        "Olomouc",
        "Ústí nad Labem",
        "Hradec Králové",
        "Pardubice",
        "Zlín",
        "České Budějovice",
        "Jihlava",
        "Karlovy Vary",
    ]
    regional_cities = [cities_dict[name] for name in regional_cities_names]

    # for all cities beginning with letter from q to z, find the closest regional city
    selected_cities = get_unprocessed_cities(
        "city_connections.json", cities_dict)
    for city in selected_cities:
        print(city)
    input("Press Enter to continue...")
    # append to the JSON file, do not overwrite
    with open("city_connections.json", "a", encoding="utf-8") as f:
        for city in selected_cities:
            try:
                closest_city, distance = Connector.find_closest_regional_city(
                    city, regional_cities
                )
                city.set_connection_to_regional_city(closest_city.name, distance)
                # dump city connection data to JSON
                json.dump(city.dump(), f, ensure_ascii=False, indent=4)
                f.write(",\n")  # Add a newline for better readability
            except Exception as e:
                f.write(f"Error processing city {city.name}: {e}")


def get_unprocessed_cities(processed_cities_file, cities_dict):
    """Return a list of cities that have not yet been processed."""
    processed_cities = set()

    #read the file via json read, the whole file is a big json which contains an array of objects
    with open(processed_cities_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

        for entry in data:
            processed_cities.add(entry["name"])

    unprocessed_cities = [
        city for name, city in cities_dict.items() if name not in processed_cities
    ]
    return unprocessed_cities

if __name__ == "__main__":
    main()
