import scrapy
from otodom.items import OtodomItem  
from urllib.parse import urljoin

class HousesSpider(scrapy.Spider):
    name = 'houses'
    allowed_domains = ["otodom.pl"]
    custom_settings = {
        "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
    }

    def start_requests(self):
        """Rozpoczyna scrapowanie od strony 1"""
        #url = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow/krakow/bienczyce?viewType=listing&page=1"
        page_count = 1    
        while page_count <= 26:
            url = f"https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/warszawa/warszawa/warszawa/bemowo?page={page_count}"
            page_count += 1
            yield scrapy.Request(url, callback=self.parse_url, meta={"proxy": "x"})

    def parse_url(self, response):
        """Zbiera linki do ofert i przechodzi do kolejnych stron"""

        # ✅ Pobieramy linki do ofert
        offers = response.css("a[data-cy='listing-item-link']::attr(href)").getall()
        self.logger.info(f"🔗 Znalezione linki: {offers}")
        for link in offers:
            full_url = urljoin(response.url, link)
            yield scrapy.Request(full_url, callback=self.parse, meta={"proxy": "x"})

        # ✅ Przechodzimy do następnej strony, jeśli istnieje
        next_page = response.css('ul[data-cy="frontend.search.base-pagination.nexus-pagination"] a[aria-label="Go to next Page"]::attr(href)').get()
        if next_page:
            next_page_url = urljoin(response.url, next_page)
            self.logger.info(f"➡️ Przechodzenie do następnej strony: {next_page_url}")
            yield scrapy.Request(next_page_url, callback=self.parse_url, meta={"proxy": "x"})

    def parse(self, response):
        """Parsuje dane z każdej oferty"""
        self.logger.info(f"Parsing URL: {response.url}")
        house_item = OtodomItem()

        # ✅ LINK
        house_item["link"] = response.url

        # ✅ CENA
        price_string = response.css('strong[data-cy="adPageHeaderPrice"]::text').get()
        if price_string:
            house_item["cena"] = int(price_string.replace(" ", "").replace("zł", ""))
        else:
            house_item["cena"] = None

        # ✅ CENA ZA METR
        price_per_meter_sqr_string = response.css('div[aria-label="Cena za metr kwadratowy"]::text').get()
        if price_per_meter_sqr_string:
            house_item["cena_za_metr_kw"] = int(price_per_meter_sqr_string.replace(" ", "").replace("zł/m²", ""))
        else:
            house_item["cena_za_metr_kw"] = None

        # ✅ LOKALIZACJA
        house_item["lokalizacja"] = response.css('div.css-pla15i.e5h9f1b2::text').get()

        # ✅ Pobieranie szczegółów oferty (ogrzewanie, piętro, itp.)
        data_info = response.css('div.css-1xw0jqp.eows69w1 p.eows69w2.css-1airkmu::text')
        data_list = [x.get() for x in data_info]

        categories = [
            "Ogrzewanie", "Piętro", "Czynsz", "Stan wykończenia", "Rynek", "Rodzaj zabudowy",
            "Liczba pięter", "D`wupoziomowe", "Forma własności", "Dostępne od",
            "Typ ogłoszeniodawcy", "Informacje dodatkowe", "Obsługa zdalna", "Garaż",
            "Miejsce Parkingowe", "Ogródek", "Taras", "Rok budowy", "Winda",
            "Materiał budynku", "Okna", "Certyfikat energetyczny", "Wyposażenie",
            "Zabezpieczenia", "Media"
        ]

        def normalize_key(text):
            """Usuwa polskie znaki i zamienia spacje na podkreślenia"""
            replacements = {
                "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
                "ó": "o", "ś": "s", "ź": "z", "ż": "z"
            }
            for pl, eng in replacements.items():
                text = text.replace(pl, eng)

            return text.lower().replace(" ", "_")

        for category in categories:
            if category in data_list:
                index = data_list.index(category)  # Znajduje indeks kategorii
                key = normalize_key(category)  # Normalizuje nazwę na wersję bez polskich znaków

                # Sprawdzamy, czy index+2 nie przekracza długości listy
                if index + 2 < len(data_list):
                    house_item[key] = data_list[index + 2]
                else:
                    house_item[key] = None  # Jeśli brak wartości, ustawiamy None

        # ✅ Pobieranie metrażu i liczby pokoi
        rooms_num = response.css('div.css-1ftqasz::text').getall()
        if len(rooms_num) > 1:
            try:
                house_item["metraz"] = float(rooms_num[0].replace("m²", "").strip())
                house_item["liczba_pokoi"] = int(rooms_num[1].split()[0])
            except ValueError:
                house_item["metraz"] = None
                house_item["liczba_pokoi"] = None

        # ✅ Pobieranie zdjęć
        house_item["zdjecia"] = response.css("img.image-gallery-thumbnail-image::attr(src)").getall()

        yield house_item
