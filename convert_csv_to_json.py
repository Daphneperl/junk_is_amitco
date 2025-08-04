import csv
import json
import os

def convert_csv_to_json(csv_file_path, json_file_path):
    """
    Convert CSV file to JSON format
    """
    data = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        # Read CSV with proper handling of quoted fields
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            # Convert string values to appropriate types
            processed_row = {}
            for key, value in row.items():
                if value == '':
                    processed_row[key] = None
                elif key in ['intimacy_level', 'rawness_percent']:
                    try:
                        processed_row[key] = float(value) if value != '-' else None
                    except ValueError:
                        processed_row[key] = None
                elif key == 'keyword_confidences':
                    # Handle the semicolon-separated confidence values
                    if value and value != '-':
                        try:
                            confidences = [float(x.strip()) for x in value.split(';')]
                            processed_row[key] = confidences
                        except ValueError:
                            processed_row[key] = None
                    else:
                        processed_row[key] = None
                elif key == 'keywords':
                    # Handle the semicolon-separated keywords
                    if value and value != '-':
                        keywords = [x.strip() for x in value.split(';')]
                        processed_row[key] = keywords
                    else:
                        processed_row[key] = None
                else:
                    processed_row[key] = value if value != '-' else None
            
            data.append(processed_row)
    
    # Write to JSON file with proper formatting
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {csv_file_path} to {json_file_path}")
    print(f"Total records: {len(data)}")

if __name__ == "__main__":
    csv_file = "assets/DF.csv"
    json_file = "assets/DF.json"
    
    if os.path.exists(csv_file):
        convert_csv_to_json(csv_file, json_file)
    else:
        print(f"Error: {csv_file} not found") 