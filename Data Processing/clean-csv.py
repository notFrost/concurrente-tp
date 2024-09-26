import csv
import codecs

# Read the clean-data.csv file
input_file = 'data-pruned.csv'
output_file = 'data-clean.csv'

columns_to_keep = ['EDAD', 'SEXO', 'FECHA_VACUNACION', 'DOSIS', 'FABRICANTE', 'REFUERZO']
sexo_mapping = {'FEMENINO': '0', 'MASCULINO': '1'}
fabricante_mapping = {
    'ASTRAZENECA': '0',
    'PFIZER': '1',
    'MODERNA': '2',
    'SINOPHARM': '3'
}

def process_file(encoding):
    with codecs.open(input_file, 'r', encoding=encoding) as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
        
        writer.writeheader()
        
        for row in reader:
            new_row = {col: row[col] for col in columns_to_keep}
            new_row['SEXO'] = sexo_mapping.get(new_row['SEXO'], new_row['SEXO'])
            new_row['FABRICANTE'] = fabricante_mapping.get(new_row['FABRICANTE'], new_row['FABRICANTE'])
            writer.writerow(new_row)

try:
    process_file('utf-8-sig')  # Try UTF-8 with BOM first
except UnicodeDecodeError:
    try:
        process_file('utf-8')  # Try UTF-8 without BOM
    except UnicodeDecodeError:
        process_file('latin-1')  # Fallback to latin-1

print(f"Data cleaning complete. New dataset saved as '{output_file}'")