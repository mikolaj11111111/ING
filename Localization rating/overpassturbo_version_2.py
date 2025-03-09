import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

data = gpd.read_file("export.geojson")
points = pd.read_csv("normalized_and_geocoded_addresses.csv")

data = data.to_crs(epsg=3857)

points = points[points['Współrzędne Geograficzne'] != 'Nie znaleziono współrzędnych'].reset_index(drop=True)
points[['x','y']] = points['Współrzędne Geograficzne'].str.split(',', expand=True)
points['x'] = points['x'].astype(float)
points['y'] = points['y'].astype(float)
points['geometry'] = gpd.points_from_xy(points['y'], points['x'], crs="EPSG:4326")
points = gpd.GeoDataFrame(points, geometry='geometry')
points = points.to_crs(epsg=3857)

# z Overpass API
categories = {
    "Sklep": data[data['shop'].notnull()],
    "Galerie handlowe": data[data['shop'] == 'mall'],
    "Przychodnia": data[data['amenity'] == 'clinic'],
    "Szpitale": data[data['amenity'] == 'hospital'],
    "Apteki": data[data['amenity'] == 'pharmacy'],
    "Przystanek Metro": data[data['railway'] == 'subway_entrance'],
    "Przystanek Autobusowy": data[data['highway'] == 'bus_stop'],
    "Przystanek Tramwajowy": data[data['railway'] == 'tram_stop'],
    "Przedszkola": data[data['amenity'] == 'kindergarten'],
    "Szkoły Podstawowe": data[data['amenity'] == 'school'],
    "Licea": data[data['amenity'] == 'college'],
    "Uniwersytety": data[data['amenity'] == 'university'],
    "Parki": data[data['leisure'] == 'park'],
    "Siłownie": data[data['leisure'] == 'fitness_centre'],
    "Basen": data[data['leisure'] == 'swimming_pool'],
    "Kościoły": data[data['amenity'] == 'place_of_worship'],
    "Poczta": data[data['amenity'] == 'post_office'],
    "Urzędy": data[data['amenity'] == 'public_building'],
    "Kino": data[data['amenity'] == 'cinema'],
    "Teatr": data[data['amenity'] == 'theatre'],
    "Filharmonia": data[data['amenity'] == 'music_venue'],
    "Opera": data[data['amenity'] == 'opera_house'],
    "Plac zabaw": data[data['leisure'] == 'playground'],
    "Dworzec główny": data[data['building'] == 'train_station']
}

def sjoin_nearest_safe(points_gdf, target_gdf, col_name):
    if not target_gdf.empty:
        target_gdf = target_gdf.reset_index(drop=True)
        joined = gpd.sjoin_nearest(points_gdf, target_gdf, how='left', distance_col=col_name)
        return joined[col_name].reset_index(drop=True)
    else:
        print(f"Brak danych dla: {col_name}")
        return pd.Series([pd.NA] * len(points_gdf), index=points_gdf.index)


for category, gdf in categories.items():
    points[category] = sjoin_nearest_safe(points, gdf, category)

# Warsaw Oldtown
starowka_point = gpd.GeoSeries([Point(21.013485, 52.247640)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
points['Starówka'] = points.distance(starowka_point)

points.to_csv("results2.csv", index=False)
points.to_excel("results2.xlsx", index=False)

