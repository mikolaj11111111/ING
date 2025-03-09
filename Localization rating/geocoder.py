import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


key = ''
geocoder_url = "https://api.opencagedata.com/geocode/v1/json"

cache = {}

def geocode_address(address):
    if address in cache:
        print(f"Korzystanie z cache dla adresu: {address}")
        return cache[address]

    print(f"Rozpoczynam geokodowanie adresu: {address}")
    params = {
        'q': address,
        'key': key,
        'limit': 1
    }
    response = requests.get(geocoder_url, params=params)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lon = data['results'][0]['geometry']['lng']
        coordinates = f"{lat}, {lon}"
        cache[address] = coordinates
        print(f"Znaleziono współrzędne dla {address}: {coordinates}")
        return coordinates
    else:
        print(f"Nie znaleziono współrzędnych dla: {address}")
        cache[address] = "Nie znaleziono współrzędnych"
        return "Nie znaleziono współrzędnych"


df = pd.read_csv('filtered_combined.csv')


with ThreadPoolExecutor(max_workers=10) as executor:
    df['Współrzędne Geograficzne'] = list(executor.map(geocode_address, df['lokalizacja']))


df.to_csv('data_coordinates.csv', index=False)
print("Dane zapisane pomyślnie.")
