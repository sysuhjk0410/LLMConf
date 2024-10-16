import pandas as pd

def convert_true_false_to_numbers(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert True and False to 1 and 0
    df = df.replace({True: 1, False: 0})
    
    # Overwrite the original file
    df.to_csv(csv_file, index=False)

convert_true_false_to_numbers('/LLMConf/Modeling/data.csv')
