import pandas as pd

# Try to import tabulate for nicer table formatting; if not available, fall back to the default print.
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

def calculate_summary_statistics(df):
    """
    Calculates summary statistics for a dataframe using the pandas describe() method.
    """
    return df.describe()

def print_summary(summary, title):
    """
    Prints a summary table with a title.
    If tabulate is available, it prints in a formatted table.
    """
    print(title)
    if tabulate:
        print(tabulate(summary, headers='keys', tablefmt='psql'))
    else:
        print(summary.to_string())
    print("\n")

def main():
    results = {}  # Dictionary to store summary DataFrames for later saving

    # --- File 1: Dzielnice Warszawy.xlsx ---
    file1 = "Dzielnice Warszawy.xlsx"
    sheet1 = "summary statistics"
    # Explicitly naming the columns expected in this dataset:
    cols1 = [
        "Bemowo", "Białołęka", "Bielany", "Mokotów", "Ochota",
        "Praga-Południe", "Praga-Północ", "Rembertów", "Targówek",
        "Ursus", "Ursynów", "Wawer", "Wesoła", "Wilanów",
        "Wola", "Włochy", "Śródmieście", "Żoliborz"
    ]
    try:
        df1 = pd.read_excel(file1, sheet_name=sheet1)
        df1 = df1[cols1]  # Subset to explicitly named columns
        summary1 = calculate_summary_statistics(df1)
        print_summary(summary1, f"Summary statistics for {file1} - Sheet: {sheet1}")
        results["Dzielnice_Warszawy"] = summary1
    except Exception as e:
        print(f"Error processing {file1} - Sheet: {sheet1}: {e}")

    # --- File 2: VAR VEC model macro.xlsx ---
    file2 = "VAR VEC model macro.xlsx"
    sheet2 = "Sheet1"
    # Explicitly naming the variables for this dataset:
    cols2 = [
        "GDP [PLN]", "lom [%]", "Inflacja do okresu w poprzednim roku",
        "PMI", "CCC", "Registered Unemployment", "Średnia cena transakcyjna za metr w WWA"
    ]
    try:
        df2 = pd.read_excel(file2, sheet_name=sheet2)
        df2 = df2[cols2]
        summary2 = calculate_summary_statistics(df2)
        print_summary(summary2, f"Summary statistics for {file2} - Sheet: {sheet2}")
        results["VAR_VEC_model_macro"] = summary2
    except Exception as e:
        print(f"Error processing {file2} - Sheet: {sheet2}: {e}")

    # --- File 3: RealEstate_WWA.xlsx ---
    file3 = "RealEstate_WWA.xlsx"
    sheet3 = "Sheet2"
    # Explicitly naming the columns expected in this dataset:
    cols3 = [
        "mieszkania oddane do użytkowania",
        "Annual growth rate of new loans to households and non-financial corporations",
        "political stimulus", "Liczba nowo podpisanych umów kredytowych",
        "Wartość nowo podp umów w mld. Zł", "Mieszkania, których budowę rozpoczęto",
        "pozwolenia wydane na budowę i zgłoszenia budowy z projektem budowlanym",
        "Średnia cena transakcyjna za metr w WWA", "20-24", "25-29",
        "30-34", "35-39", "40-44"
    ]
    try:
        df3 = pd.read_excel(file3, sheet_name=sheet3)
        df3 = df3[cols3]
        summary3 = calculate_summary_statistics(df3)
        print_summary(summary3, f"Summary statistics for {file3} - Sheet: {sheet3}")
        results["RealEstate_WWA"] = summary3
    except Exception as e:
        print(f"Error processing {file3} - Sheet: {sheet3}: {e}")

    # --- Saving All Summaries to a Single Excel File ---
    output_file = "Summary_statistics.xlsx"
    try:
        with pd.ExcelWriter(output_file) as writer:
            for sheet_name, summary_df in results.items():
                # Excel sheet names have a maximum length of 31 characters.
                writer_sheet = sheet_name[:31]
                summary_df.to_excel(writer, sheet_name=writer_sheet)
        print(f"Summary statistics have been saved to {output_file}")
    except Exception as e:
        print(f"Error saving summary statistics to {output_file}: {e}")

if __name__ == '__main__':
    main()
