"""A module containing static data used across the project."""

REGIONAL_CITIES_NAMES = [
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

LARGE_CITIES_NAMES = [
    "Praha",
    "Brno",
    "ostrava"
]

CAPITAL = "Praha"

NUTS_OKRES_CODES = [
    "CZ0100",  # Praha
    # Středočeský kraj
    "CZ0201",
    "CZ0202",
    "CZ0203",
    "CZ0204",
    "CZ0205",
    "CZ0206",
    "CZ0207",
    "CZ0208",
    "CZ0209",
    "CZ020A",
    "CZ020B",
    "CZ020C",
    # Jihočeský kraj
    "CZ0311",
    "CZ0312",
    "CZ0313",
    "CZ0314",
    "CZ0315",
    "CZ0316",
    "CZ0317",
    # Plzeňský kraj
    "CZ0321",
    "CZ0322",
    "CZ0323",
    "CZ0324",
    "CZ0325",
    "CZ0326",
    "CZ0327",
    # Karlovarský kraj
    "CZ0411",
    "CZ0412",
    "CZ0413",
    # Ústecký kraj
    "CZ0421",
    "CZ0422",
    "CZ0423",
    "CZ0424",
    "CZ0425",
    "CZ0426",
    "CZ0427",
    # Liberecký kraj
    "CZ0511",
    "CZ0512",
    "CZ0513",
    "CZ0514",
    # Královéhradecký kraj
    "CZ0521",
    "CZ0522",
    "CZ0523",
    "CZ0524",
    "CZ0525",
    # Pardubický kraj
    "CZ0531",
    "CZ0532",
    "CZ0533",
    "CZ0534",
    # Kraj Vysočina
    "CZ0631",
    "CZ0632",
    "CZ0633",
    "CZ0634",
    "CZ0635",
    # Jihomoravský kraj
    "CZ0641",
    "CZ0642",
    "CZ0643",
    "CZ0644",
    "CZ0645",
    "CZ0646",
    "CZ0647",
    # Olomoucký kraj
    "CZ0711",
    "CZ0712",
    "CZ0713",
    "CZ0714",
    "CZ0715",
    # Zlínský kraj
    "CZ0721",
    "CZ0722",
    "CZ0723",
    "CZ0724",
    # Moravskoslezský kraj
    "CZ0801",
    "CZ0802",
    "CZ0803",
    "CZ0804",
    "CZ0805",
    "CZ0806",
]


ELECTION_BASE_URL = "https://www.volby.cz/appdata/ps2025/odata/okresy/vysledky_okres_{nuts}.xml"
XML_NS = {"ps": "http://www.volby.cz/ps/"}