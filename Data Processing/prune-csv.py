import csv
from tqdm import tqdm

INPUT_FILE = "data-preprocessed.csv"
OUTPUT_FILE = "data-pruned.csv"
MAX_ROWS = 1_000_000

def clean_csv():
    with open(INPUT_FILE, 'r', newline='', encoding='utf-8') as input_file, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as output_file:
        
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Read and modify the header row
        header = next(reader)
        fecha_vacunacion_index = header.index("FECHA_VACUNACION")
        dosis_index = header.index("DOSIS")
        header.append("REFUERZO")
        writer.writerow(header)

        # Write up to 1 million data rows that meet the criteria
        rows_written = 0
        for row in tqdm(reader, desc="Processing rows"):
            try:
                # Check if FECHA_VACUNACION starts with "2021"
                if row[fecha_vacunacion_index].startswith("2021"):
                    # Add REFUERZO column
                    fecha = row[fecha_vacunacion_index]
                    dosis = row[dosis_index]
                    refuerzo = "1" if (dosis < "3" and fecha.startswith(("202101", "202102", "202103", "202104", "202105", "202106"))) else "0"
                    row.append(refuerzo)
                    
                    writer.writerow(row)
                    rows_written += 1
                    if rows_written >= MAX_ROWS:
                        break
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

if __name__ == "__main__":
    clean_csv()
    print(f"Cleaned CSV file with up to {MAX_ROWS} rows from 2021, including REFUERZO column, has been created: {OUTPUT_FILE}")