"""
===========================================================================
Excel Combiner
===========================================================================

Purpose
-------
This script combines all Excel (.xlsx) files located in the same folder
into a single Excel file.

Instead of creating multiple sheets, it appends all rows into one sheet.

Requirements
------------
- Python 3.x
- pandas
- openpyxl

Install required packages:
    pip install pandas openpyxl

Folder Structure
----------------
Place this Python file in the same folder as all Excel files.

Example:

    Project/
    │
    ├── combine_excel.py
    ├── File1.xlsx
    ├── File2.xlsx
    ├── File3.xlsx
    └── ...

How to Run
----------
Open a terminal in this folder and execute:

    python combine_excel.py

Output
------
A new file named:

    combined_excel.xlsx

will be created in the same folder.

The output contains:
- One worksheet only
- All rows from every Excel file
- Original column names preserved
- Rows appended in the order the files are read

Notes
-----
- All input Excel files should have the same column structure.
- Existing "combined_excel.xlsx" is ignored to prevent duplicate data.
- Original Excel files are never modified.

===========================================================================
"""

import os
import glob
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))

excel_files = glob.glob(os.path.join(folder, "*.xlsx"))

combined_data = []

for file in excel_files:
    # Skip previously generated output file
    if os.path.basename(file) == "combined_excel.xlsx":
        continue

    df = pd.read_excel(file)
    combined_data.append(df)

if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True)

    output_file = os.path.join(folder, "combined_excel.xlsx")
    final_df.to_excel(output_file, index=False)

    print("=" * 60)
    print("Excel files combined successfully.")
    print(f"Files Processed : {len(combined_data)}")
    print(f"Total Rows      : {len(final_df)}")
    print(f"Output File     : {output_file}")
    print("=" * 60)

else:
    print("No Excel (.xlsx) files found in this folder.")
