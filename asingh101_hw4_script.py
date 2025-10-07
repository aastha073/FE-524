#!/usr/bin/env python3
"""
PDF Table Extraction to CSV using OpenAI API
FE524-A Homework 4
"""

import os
import pymupdf  # PyMuPDF
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_table_to_csv(pdf_text, pdf_filename):
    """Use OpenAI API to extract table data and convert to CSV"""
    
    prompt = f"""You are extracting structured data table information from a PDF document about database file specifications.

Extract the table that describes the data structure/fields. The table should include these columns:
- file_name (the filename mentioned at the top of the section)
- key (the key identifier like A#1, B#2, etc.)
- item (the field name)
- data_type (X for character, N for numeric, D for decimal)
- format (the format specification)
- length (number of characters)
- start (starting position)
- end (ending position)
- comments (any additional notes)

Source PDF filename: {pdf_filename}

PDF Content:
{pdf_text}

Return ONLY the CSV data with a header row. Each row should be a field from the table. Format the CSV properly with commas separating values. If a field is empty, leave it blank but include the comma.

Example format:
file_name,key,item,data_type,format,length,start,end,comments
EXLFILAT.DAT,A#1,I/B/E/S Ticker,X,CCCCCC,6,1,6,
EXLFILAT.DAT,B#2,Estimator,N,99999,5,8,12,

Extract all rows from the table in the PDF."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data extraction expert. Extract table data exactly as specified and return only CSV format with no additional text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000
    )
    
    return response.choices[0].message.content.strip()

def main():
    # PDF files to process
    pdf_files = [
        "ibes_detail_history_docs_13.pdf",
        "ibes_detail_history_docs_15.pdf", 
        "ibes_summary_history_docs_14.pdf"
    ]
    
    # Output CSV files
    output_files = [
        "ibes_detail_history_docs_13.csv",
        "ibes_detail_history_docs_15.csv",
        "ibes_summary_history_docs_14.csv"
    ]
    
    for pdf_file, output_file in zip(pdf_files, output_files):
        print(f"Processing {pdf_file}...")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        
        # Extract table and convert to CSV
        csv_content = extract_table_to_csv(pdf_text, pdf_file)
        
        # Save to CSV file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        print(f"  -> Saved to {output_file}")
    
    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()
