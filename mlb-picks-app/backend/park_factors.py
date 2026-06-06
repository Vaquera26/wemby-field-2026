"""
Park factors MLB — calibrados con promedios historicos 2020-2025.
1.0 = neutro. >1.0 = mas carreras. <1.0 = menos carreras.
Fuente: Baseball Reference park factors (runs, 3yr rolling average).
"""

PARK_FACTORS: dict[str, float] = {
    # Alta anotacion
    "Coors Field":                1.19,   # Denver altitud — el mas alto de la liga
    "Great American Ball Park":   1.08,   # Cincinnati
    "Fenway Park":                1.06,   # Boston — monstruo verde corto
    "Kauffman Stadium":           1.04,
    "Guaranteed Rate Field":      1.04,   # Chicago White Sox
    "Citizens Bank Park":         1.04,   # Philadelphia
    "Truist Park":                1.02,   # Atlanta
    "Minute Maid Park":           1.02,
    "Daikin Park":                1.02,

    # Neutro
    "Yankee Stadium":             1.01,
    "Wrigley Field":              1.00,   # viento lo hace variable
    "Busch Stadium":              1.00,
    "Target Field":               0.99,
    "American Family Field":      0.99,
    "Nationals Park":             0.99,
    "Angel Stadium":              0.99,
    "Globe Life Field":           0.99,
    "Rogers Centre":              0.98,
    "Comerica Park":              0.98,
    "Progressive Field":          0.98,
    "PNC Park":                   0.97,
    "Citi Field":                 0.97,

    # Suprime carreras
    "T-Mobile Park":              0.96,   # Seattle — noche/humedad
    "Chase Field":                0.96,
    "loanDepot park":             0.96,
    "LoanDepot Park":             0.96,
    "Dodger Stadium":             0.95,
    "Petco Park":                 0.94,   # San Diego — clima marino
    "Oracle Park":                0.93,   # San Francisco — viento marino
    "Oakland Coliseum":           0.93,
    "Oriole Park at Camden Yards":0.97,
    "Tropicana Field":            0.97,
}

def get_park_factor(venue_name: str) -> float:
    """Devuelve park factor para un estadio. Default 1.00 si no se encuentra."""
    if not venue_name:
        return 1.00
    vl = venue_name.lower()
    for name, factor in PARK_FACTORS.items():
        if name.lower() in vl or vl in name.lower():
            return factor
    return 1.00
