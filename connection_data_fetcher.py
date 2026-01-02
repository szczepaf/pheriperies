"""This module essentially creates the dataset.

It fetches connection data using the connector class. It then fetches the election data using beatifulsoup4 and xml parsing.
It saves the connection data to city_connections_verbose.json and election data to election_data.csv in the db folder."""

import json
import sys
import math
import db.static as static

import pandas as pd
from city import City
from connector import Connector
from concurrent.futures import ThreadPoolExecutor, as_completed


import xml.etree.ElementTree as ET
from typing import Iterable, Union

import requests

sys.stdout.reconfigure(encoding="utf-8")



def fetch_cities_dict_with_population_above_threshold(threshold: int = 10000) -> dict:
    """Return a dict of cities with population at least 10,000 (or a given threshold)."""
    df = pd.read_csv("db\population_data.csv", index_col=False)

    # Convert population to numeric; invalid / empty values become NaN
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    # Filter municipalities with population at least 10,000
    df_10k = df[df["population"] >= threshold].copy()
    # create a dict where cities are queryable by name
    cities_dict = {}

    for _, row in df_10k.iterrows():
        name = str(row["municipality_name"])
        population = int(row["population"])

        city = City(name, population)
        # regional_city and distance already defaulted in __init__
        cities_dict[name] = city

    return cities_dict


def _connection_worker_for_concurrency(city: City, regional_cities: list[City]) -> dict:
    """Concurrency helper."""
    try:
        Connector.find_connections_to_all_regional_cities(city, regional_cities)
        return city.verbose_dump()
    except Exception as e:
        # Optional: keep a record that this city failed
        return {
            "name": city.name,
            "population": city.population,
            "connections": {},
            "error": str(e),
        }

def fetch_connections(target_file: str = "db/city_connections_verbose.json"):
    """Fetch connection data for all cities with population above threshold
    to all regional cities, and save to city_connections_verbose.json
    as a single valid JSON object: {"data": [...]}.
    """

    # 1) Load all cities (population >= threshold)
    cities_dict = fetch_cities_dict_with_population_above_threshold()

    # 2) Load existing JSON store or initialise a new one
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            store = json.load(f)
            if not isinstance(store, dict):
                store = {"data": []}
    except (FileNotFoundError, json.JSONDecodeError):
        store = {"data": []}

    if "data" not in store or not isinstance(store["data"], list):
        store["data"] = []

    # 3) Build list of regional City objects
    regional_cities = [cities_dict[name] for name in static.REGIONAL_CITIES_NAMES]

    # 4) Determine which cities still need processing
    selected_cities = get_unprocessed_cities(target_file, cities_dict)
    if not selected_cities:
        print("No unprocessed cities found – nothing to do.")
        return

    print(f"Processing {len(selected_cities)} cities...")

    # 5) Compute connections in parallel (I/O-bound → threads are fine)
    new_records: list[dict] = []
    max_workers = 4
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_city = {
                executor.submit(_connection_worker_for_concurrency, city, regional_cities): city
                for city in selected_cities
            }

            for future in as_completed(future_to_city):
                city = future_to_city[future]
                print(f"Processed city: {city.name}")
                try:
                    result = future.result()
                    new_records.append(result)
                except Exception as e: 
                    # This should be rare because _compute_connections_for_city catches most issues
                    print(f"Unexpected error for city {city.name}: {e}")

        # 6) Extend existing data and rewrite the file
        store["data"].extend(new_records)
    finally: # always write
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=4)

    print(f"Added {len(new_records)} cities to {target_file}.")



def get_unprocessed_cities(processed_cities_file, cities_dict):
    """Return a list of cities that have not yet been processed."""
    processed_cities = set()

    try:
        with open(processed_cities_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
            data = payload.get("data", [])
    except (FileNotFoundError, json.JSONDecodeError):
        # First run or corrupted file → treat as no processed cities
        data = []

    for entry in data:
        processed_cities.add(entry["name"])

    unprocessed_cities = [
        city for name, city in cities_dict.items() if name not in processed_cities
    ]
    return unprocessed_cities


def clear_up_connections_db(target_file: str = "db/city_connections_verbose.json") -> None:
    """Remove all city records that contain an 'error' key from the connections JSON."""
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            store = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Nothing to clean or invalid file → just return
        return

    data = store.get("data", [])

    cleaned = [rec for rec in data if not (isinstance(rec, dict) and "error" in rec)]
    
    store["data"] = cleaned
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=4)

def repair_infinity_distances_in_connections_db(
    target_file: str = "db/city_connections_verbose.json",
    batch_size: int = 10,
) -> None:
    """
    Scan the connections JSON file and, for up to `batch_size` cities that contain
    at least one infinity value in their `connections` dict, recompute those
    specific connections using Connector.find_connection and overwrite them.

    The JSON is expected to have the structure:
        {"data": [ { "name": ..., "population": ..., "connections": {...} }, ... ]}
    """

    # Load the existing store
    with open(target_file, "r", encoding="utf-8") as f:
        store = json.load(f)

    data = store.get("data", [])
    repaired_cities = 0

    for record in data:
        if repaired_cities >= batch_size:
            break

        connections = record.get("connections")
        if not isinstance(connections, dict):
            continue  # skip invalid records

        # Find which destinations have infinite distances
        inf_destinations = []
        for dest_name, dist in connections.items():
            is_inf = False
            if isinstance(dist, (int, float)):
                is_inf = math.isinf(dist)
            elif isinstance(dist, str):
                # Extra robustness if someone ever wrote "inf"/"Infinity" as a string
                if dist.strip().lower() in ("inf", "infinity"):
                    is_inf = True

            if is_inf:
                inf_destinations.append(dest_name)

        if not inf_destinations:
            continue  # this city has no infinities

        # Rebuild City objects for origin and destination(s)
        city_name = record.get("name")
        population = record.get("population", 0)
        origin_city = City(city_name, population)

        for dest_name in inf_destinations:
            # For Connector.find_connection, only the .name matters
            dest_city = City(dest_name, 0)
            try:
                new_distance = Connector.find_connection(
                    origin_city, dest_city, Connector.SELECTED_DATE
                )
                connections[dest_name] = new_distance
            except Exception as e:
                # If recomputation fails, leave the original value in place
                print(f"Failed to repair connection {city_name} -> {dest_name}: {e}")

        repaired_cities += 1

    # Write the updated store back
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=4)

def _normalise_city_name(name: str) -> str:
    """
    Simple normalisation helper: strip and lower.
    (Keeps diacritics, only case-insensitive.)
    """
    return name.strip().lower()


def fetch_and_save_election_data_for_cities(
    cities: Iterable[Union[str, "City"]],
    output_csv: str = "db\election_data.csv",
) -> pd.DataFrame:
    """
    Download parliamentary election results for the given list of cities
    and save them into a CSV file.

    Parameters
    ----------
    cities : iterable of str or City
        Names of municipalities to keep. If City objects are passed,
        their .name attribute is used.
    output_csv : str
        Path where the resulting CSV will be written.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per city and columns:
        - city_name
        - ZAPSANI_VOLICI
        - UCAST_PROC
        - for parties 1..26: P{N}_HLASY, P{N}_PROC
    """
    requested_names = set()
    for c in cities:
        if isinstance(c, str):
            requested_names.add(_normalise_city_name(c))
        else:
            # assume City-like object with .name
            requested_names.add(_normalise_city_name(c.name))

    rows = []

    for nuts in NUTS_OKRES_CODES:
        url = static.BASE_URL.format(nuts=nuts)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            # You can log this if you like; for now just skip this okres
            print(f"Warning: failed to download {url}: {e}")
            continue

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            print(f"Warning: failed to parse XML for {url}: {e}")
            continue

        # Iterate over all municipalities (OBEC elements)
        for obec in root.findall(".//ps:OBEC", static.XML_NS):
            typ = obec.attrib.get("TYP_OBEC")
            # Only OBEC_BEZ_MCMO and OBEC_S_MCMO
            if typ not in ("OBEC_BEZ_MCMO", "OBEC_S_MCMO"):
                continue

            city_name = obec.attrib.get("NAZ_OBEC", "").strip()
            if _normalise_city_name(city_name) not in requested_names:
                continue

            ucast = obec.find("ps:UCAST", static.XML_NS)
            if ucast is None:
                # No turnout info, skip (should not happen)
                continue

            row = {
                "city_name": city_name,
                "ZAPSANI_VOLICI": int(ucast.attrib.get("ZAPSANI_VOLICI", "0")),
                "UCAST_PROC": float(
                    ucast.attrib.get("UCAST_PROC", "0").replace(",", ".")
                ),
            }

            # Initialise all 26 party columns with zeros
            for k in range(1, 27):
                row[f"P{k}_HLASY"] = 0
                row[f"P{k}_PROC"] = 0.0

            # Fill in actual values from HLASY_STRANA
            for hs in obec.findall("ps:HLASY_STRANA", static.XML_NS):
                kstrana = hs.attrib.get("KSTRANA")
                if not kstrana:
                    continue

                try:
                    k = int(kstrana)
                except ValueError:
                    continue

                if 1 <= k <= 26:
                    hlasy = int(hs.attrib.get("HLASY", "0"))
                    proc = float(hs.attrib.get("PROC_HLASU", "0").replace(",", "."))
                    row[f"P{k}_HLASY"] = hlasy
                    row[f"P{k}_PROC"] = proc

            rows.append(row)

    if not rows:
        print("No matching cities found in downloaded election data.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure consistent column order: city, turnout, then parties 1..26 (abs + rel)
    columns = ["city_name", "ZAPSANI_VOLICI", "UCAST_PROC"]
    for k in range(1, 27):
        columns.append(f"P{k}_HLASY")
        columns.append(f"P{k}_PROC")

    # Add any missing columns (in case future changes add more keys to rows)
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]
    # keep only the largest occurrence of each city (by ZAPSANI_VOLICI) ---
    # print cities with duplicates for debugging
    duplicate_cities = df[df.duplicated(subset=["city_name"], keep=False)]
    print("Duplicate cities found in election data (in the format city : count):")
    print(duplicate_cities["city_name"].value_counts().to_dict())
    # now keep only the one with largest ZAPSANI_VOLICI
    df = df.sort_values(["city_name", "ZAPSANI_VOLICI"], ascending=[True, False])
    df = df.groupby("city_name", as_index=False).head(1)

    # Save to CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    # # If you have a dict of City objects as before:
    # cities_dict = create_cities_dict()
    # selected_cities = list(cities_dict.keys())  # or ú subset

    # df_elections = fetch_and_save_election_data_for_cities(selected_cities)
    # # election_data.csv is now written to disk
    
    fetch_connections()