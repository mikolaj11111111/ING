import pandas as pd

def calculate_rating(file_path, output_path):

    df = pd.read_excel(file_path)
    
    categories = {
        "Transport": 0.30,
        "Edukacja": 0.20,
        "Zdrowie": 0.15,
        "Handel": 0.10,
        "Czas wolny": 0.15,
        "Usługi publiczne": 0.10
    }
    
    category_columns = {
        "Transport": ["Metro", "Autobusy", "Tramwaje", "Dworzec główny"],
        "Edukacja": ["Przedszkola", "Szkoły", "Licea", "Uniwersytety"],
        "Zdrowie": ["Szpitale", "Przychodnie", "Apteki"],
        "Handel": ["Sklepy", "Galerie"],
        "Czas wolny": ["Parki", "Siłownie", "Baseny", "Kino", "Teatr", "Filharmonia", "Opera"],
        "Usługi publiczne": ["Kościoły", "Urzędy", "Poczta", "Plac zabaw", "Starówka"]
    }
    
#wzor z metydologii
    df["rating"] = 0
    for category, weight in categories.items():
        for col in category_columns[category]:
            if col in df.columns:
                df["rating"] += weight * (1 / (1 + df[col]))

    df.to_excel(output_path, index=False)
    
    print(f"Plik zapisany: {output_path}")


calculate_rating("filtered_results2.xlsx", "filtered_results2_with_rating.xlsx")
