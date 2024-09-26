import csv
from tqdm import tqdm

INPUT_FILE = "data-raw.csv"
PREPROCESSED_FILE = "data-preprocessed.csv"

def preprocess_csv():
    with open(INPUT_FILE, 'r', newline='', encoding='utf-8', errors='ignore') as input_file, \
         open(PREPROCESSED_FILE, 'w', newline='', encoding='utf-8') as output_file:
        
        reader = csv.reader(input_file, delimiter=';')
        writer = csv.writer(output_file, delimiter=',')

        # Get total number of rows for the progress bar
        total_rows = sum(1 for row in input_file)
        input_file.seek(0)

        for row in tqdm(reader, total=total_rows, desc="Preprocessing"):
            writer.writerow(row)

    print(f"Preprocessed CSV file has been created: {PREPROCESSED_FILE}")

if __name__ == "__main__":
    preprocess_csv()