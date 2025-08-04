import json
import pandas as pd
import csv

def merge_artistic_analysis():
    # Load the artistic analysis JSON data
    with open('image_analysis/images2_analysis/artistic_analysis_images2_filtered.json', 'r') as f:
        artistic_data = json.load(f)
    
    # Load the existing CSV dataset
    df = pd.read_csv('assets/Junk_isDataSet.csv')
    
    # Create a dictionary to store artistic analysis data by filename
    artistic_dict = {}
    for item in artistic_data:
        filename = item['filename']
        artistic_dict[filename] = {
            'description': item.get('description', ''),
            'keywords': item.get('keywords', []),
            'vibe': item.get('vibe', '')
        }
    
    # Add new columns to the dataframe
    df['artistic_description'] = ''
    df['keywords'] = ''
    df['keyword_confidences'] = ''
    df['vibe'] = ''
    
    # Merge the data
    for index, row in df.iterrows():
        filename = row['filename']
        if filename in artistic_dict:
            artistic_info = artistic_dict[filename]
            
            # Add description
            df.at[index, 'artistic_description'] = artistic_info['description']
            
            # Add vibe
            df.at[index, 'vibe'] = artistic_info['vibe']
            
            # Process keywords and confidences
            keywords = artistic_info['keywords']
            if keywords:
                keyword_list = [kw['keyword'] for kw in keywords]
                confidence_list = [str(kw['confidence']) for kw in keywords]
                
                df.at[index, 'keywords'] = '; '.join(keyword_list)
                df.at[index, 'keyword_confidences'] = '; '.join(confidence_list)
    
    # Save the merged dataset
    output_filename = 'assets/Junk_isDataSet_with_artistic_analysis.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"Merged dataset saved to: {output_filename}")
    print(f"Total records: {len(df)}")
    print(f"Records with artistic analysis: {len(df[df['keywords'] != ''])}")
    
    # Show some statistics
    print("\nVibe distribution:")
    vibe_counts = df['vibe'].value_counts()
    for vibe, count in vibe_counts.items():
        if vibe:  # Skip empty values
            print(f"  {vibe}: {count}")
    
    # Show sample of merged data
    print("\nSample of merged data:")
    sample_cols = ['filename', 'title', 'vibe', 'keywords', 'keyword_confidences']
    print(df[sample_cols].head(10).to_string())

if __name__ == "__main__":
    merge_artistic_analysis() 