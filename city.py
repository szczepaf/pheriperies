class City:
    def __init__(self, name, population):
        self.name = name
        self.population = population

    def set_connection_to_regional_city(self, regional_city, distance):
        self.regional_city = regional_city
        self.distance = distance

    def __str__(self):
        "a json like string representation of the city"
        return f'{{"name": "{self.name}", "population": {self.population}}}'

    def raw_dump(self):
        return {
            "name": self.name,
            "population": self.population,
        }

    def dump(self):
        return {
            "name": self.name,
            "population": self.population,
            "regional_city": self.regional_city,
            "distance": self.distance,
        }

    def load_from_connections_dump(self, dump: dict):
        name = dump.get("name")
        population = dump.get("population")
        regional_city = dump.get("regional_city")
        distance = dump.get("distance")

        # call the constructor first
        self.__init__(name, population)
        self.set_connection_to_regional_city(regional_city, distance)


