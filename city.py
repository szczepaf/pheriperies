class City:
    """A class representing a city with its name, population, and connections to regional cities."""

    def __init__(self, name, population):
        """Init via name of the city and its population. Create an empty dict for the regionnal city connections."""
        self.name = name
        self.population = population
        self.connections = {}

    def set_connection_to_regional_city(self, regional_city, distance):
        """Set the connection to the closest regional city along with distance to it"""
        self.regional_city = regional_city
        self.distance = distance

    def set_connections_to_all_regional_cities(self, connections: dict):
        """Set the connections to all regional cities along with distances to them.
        The expected format of the dict is: {regional_city_name: distance, ...}"""
        self.connections = connections

    def __str__(self):
        "a json like string representation of the city"
        return f'{{"name": "{self.name}", "population": {self.population}}}'

    def raw_dump(self):
        """A raw dump of the city without connection info"""
        return {
            "name": self.name,
            "population": self.population,
        }

    def dump(self):
        """Dump include connection info"""
        return {
            "name": self.name,
            "population": self.population,
            "regional_city": self.regional_city,
            "distance": self.distance,
        }

    def verbose_dump(self):
        """A verbose dump including all connections to regional cities"""
        return {
            "name": self.name,
            "population": self.population,
            "connections": self.connections,
        }

    def load_from_connections_dump(self, dump: dict):
        """Load the city from a dump including connection info"""
        name = dump.get("name")
        population = dump.get("population")
        regional_city = dump.get("regional_city")
        distance = dump.get("distance")

        self.__init__(name, population)
        self.set_connection_to_regional_city(regional_city, distance)

    def load_from_verbose_dump(self, dump: dict):
        """Load the city from a verbose dump including all connections to regional cities"""
        name = dump.get("name")
        population = dump.get("population")
        connections = dump.get("connections")

        self.__init__(name, population)
        self.set_connections_to_all_regional_cities(connections)
