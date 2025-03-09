import csv

class CsvPipeline:
    def open_spider(self, spider):
        self.file = open('bemowo.csv', 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "link", "cena", "cena_za_metr_kw", "lokalizacja", "liczba_pokoi", "metraz",
            "ogrzewanie", "pietro", "czynsz", "stan_wykonczenia", "rynek", "rodzaj_zabudowy",
            "liczba_pieter", "dwupoziomowe", "forma_wlasnosci", "dostepne_od",
            "typ_ogloszeniodawcy", "informacje_dodatkowe", "obsluga_zdalna",
            "garaz", "miejsce_parkingowe", "ogrodek", "taras", "rok_budowy",
            "winda", "material_budynku", "okna", "certyfikat_energetyczny",
            "wyposazenie", "zabezpieczenia", "media", "zdjecia"
        ])

    def process_item(self, item, spider):
        self.writer.writerow([
            item.get("link", ""),
            item.get("cena", ""),
            item.get("cena_za_metr_kw", ""),
            item.get("lokalizacja", ""),
            item.get("liczba_pokoi", ""),
            item.get("metraz", ""),
            item.get("ogrzewanie", ""),
            item.get("pietro", ""),
            item.get("czynsz", ""),
            item.get("stan_wykonczenia", ""),
            item.get("rynek", ""),
            item.get("rodzaj_zabudowy", ""),
            item.get("liczba_pieter", ""),
            item.get("dwupoziomowe", ""),
            item.get("forma_wlasnosci", ""),
            item.get("dostepne_od", ""),
            item.get("typ_ogloszeniodawcy", ""),
            "; ".join(item.get("informacje_dodatkowe", [])),
            item.get("obsluga_zdalna", ""),
            item.get("garaz", ""),
            item.get("miejsce_parkingowe", ""),
            item.get("ogrodek", ""),
            item.get("taras", ""),
            item.get("rok_budowy", ""),
            item.get("winda", ""),
            item.get("material_budynku", ""),
            item.get("okna", ""),
            item.get("certyfikat_energetyczny", ""),
            item.get("wyposazenie", ""),
            item.get("zabezpieczenia", ""),
            item.get("media", ""),
            "; ".join(item.get("zdjecia", []))
        ])
        return item

    def close_spider(self, spider):
        self.file.close()
