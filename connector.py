import requests
from bs4 import BeautifulSoup
from city import City
import time



class Connector:
    """A class for finding connections between cities using the IDOS service."""
    URL_TEMPLATE = """https://idos.idnes.cz/vlaky/spojeni/?date=DATE_PARAM&time=TIME_PARAM&f=ORIGIN_CITY_NAME&t=DESTINATION_CITY_NAME&submit=true"""

    # Every result page returns only three results, so whenever we query, we query multiple times to cover the day
    TIMES = [
            "04:00",
            "08:00",
            "12:00",
            "16:00",
            "20:00",
        ]

    SELECTED_DATE = "1. 12. 2025" # an arbitrary workday

    @staticmethod
    def find_connection(
        city_origin: City, city_destination: City, date: str
    ) -> float:
        """
        Find the best connection time (in minutes) between two cities on a given date.

        Parameters:
            city_origin (City): The origin city.
            city_destination (City): The destination city.
            date (str): The date of travel in the format "DD.MM.YYYY".

        Returns:
            float: The best connection time in minutes. Returns float('inf') if no connection is found.
        """
        if city_origin == city_destination:
            return 0
        best_connection_time = float("inf")

        for time_str in Connector.TIMES:
            url: str = (
                Connector.URL_TEMPLATE.replace("ORIGIN_CITY_NAME", city_origin.name)
                .replace("DESTINATION_CITY_NAME", city_destination.name)
                .replace("TIME_PARAM", time_str)
                .replace("DATE_PARAM", date)
            )

            response = requests.get(url)
            response.raise_for_status()  # Raise for HTTP errors
            time.sleep(1)  # Be nice to the server
            parsed_data: dict = Connector.parse_connection_data(response.text)
            current_best: float = parsed_data["best_connection"]

            if current_best < best_connection_time:
                best_connection_time = current_best

        return best_connection_time

    @staticmethod
    def find_closest_regional_city(city: City, regional_cities: list[City]) -> tuple[City, float]:
        """From a list of regional cities, find the one with the shortest connection to the given city.
        
        Return the closest regional city and the connection time in minutes."""
        if city in regional_cities:
            return city, 0.0

        closest_city = None
        min_distance = float("inf")
        for regional_city in regional_cities:
            distance = Connector.find_connection(city, regional_city, Connector.SELECTED_DATE)
            if distance < min_distance:
                min_distance = distance
                closest_city = regional_city
        return closest_city, min_distance
    
    @staticmethod
    def find_connections_to_all_regional_cities(city: City, regional_cities: list[City]):
        """Set the connections to all regional cities along with distances to them."""
        connections = {}
        for regional_city in regional_cities:
            distance = Connector.find_connection(city, regional_city, Connector.SELECTED_DATE)
            connections[regional_city.name] = distance
        city.set_connections_to_all_regional_cities(connections)
        

    @staticmethod
    def _parse_duration_to_minutes(duration_text: str) -> int:
        """
        Parse strings like:
            '41 min'
            '1 hod 2 min'
        into total minutes (int).
        """
        duration_text = duration_text.strip()
        tokens = duration_text.split()

        minutes = 0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.isdigit():
                value = int(tok)
                unit = tokens[i + 1] if i + 1 < len(tokens) else ""
                unit = unit.lower()

                if unit.startswith("hod"):  # hours
                    minutes += value * 60
                    i += 2
                elif unit.startswith("min"):  # minutes
                    minutes += value
                    i += 2
                else:
                    # Fallback: treat as minutes if unit is unknown
                    minutes += value
                    i += 1
            else:
                i += 1

        return minutes

    @staticmethod
    def parse_connection_data(data: str):
        """
        Parse the HTML of an IDOS result page, find the shortest connection
        (by 'Celkový čas') and return its basic info.
        """

        soup = BeautifulSoup(data, "html.parser")
        connection_list = soup.find("div", class_="connection-list")

        # Default "no connection" result
        best = {
            "best_connection": float("inf"),  # duration in minutes
            "from": None,
            "to": None,
            "departure_time": None,
            "arrival_time": None,
            "duration_text": None,
        }
        if connection_list is None:
            return best

        # Each connection is in a box with class "connection"
        for box in connection_list.select("div.box.connection"):
            head = box.find("div", class_="connection-head")
            if not head:
                continue

            # Celkový čas <strong>...</strong>
            total_p = head.find("p", class_="total") or head.find("p", class_="reset total")
            if not total_p:
                continue

            strong = total_p.find("strong")
            if not strong:
                continue

            duration_text = strong.get_text(strip=True)
            duration_minutes = Connector._parse_duration_to_minutes(duration_text)

            # If this connection is not better than current best, skip
            if duration_minutes >= best["best_connection"]:
                continue

            # Try to extract from/to station names and times
            stations = box.select("ul.stations li.item")
            from_name = to_name = dep_time = arr_time = None

            if len(stations) >= 2:
                dep = stations[0]
                arr = stations[-1]

                dep_time_tag = dep.find("p", class_="time")
                arr_time_tag = arr.find("p", class_="time")
                from_name_tag = dep.find("strong", class_="name")
                to_name_tag = arr.find("strong", class_="name")

                dep_time = dep_time_tag.get_text(strip=True) if dep_time_tag else None
                arr_time = arr_time_tag.get_text(strip=True) if arr_time_tag else None
                from_name = (
                    from_name_tag.get_text(strip=True) if from_name_tag else None
                )
                to_name = to_name_tag.get_text(strip=True) if to_name_tag else None

            # Update best
            best = {
                "best_connection": duration_minutes,
                "from": from_name,
                "to": to_name,
                "departure_time": dep_time,
                "arrival_time": arr_time,
                "duration_text": duration_text,
            }

        return best
