---
{
  "title": "Using Python to import multiple Excel files and export as one text file",
  "subtitle": "How I imported many Excel files into Python and then exported one text file",
  "date": "2018-03-12",
  "slug": "excel-to-python-to-text",
  "tags": ["Python", "Excel"]
}
---
<!--more-->

### How I imported many Excel files into Python and then exported one text file

At my previous job, some clients for projects provided 90 to 100 Excel files that I needed to import into SQL Server or somewhere else. To do that I wanted to create one text file that could be imported into something else. In this case it required a little more work because the files had dates and times in their names but the date wasn't in the data. I wanted to keep the date though, so I added a field for each file that contained the date and time part of the filename. I thought Python would be good for everything I wanted to do and it was. Here's what I did.

First I imported the necessary libraries.


```python
# Imports
import os
import pandas as pd
import glob
```

Then I changed to the directory that contained the workbooks and put all of the filenames in a list.


```python
# Change directory
os.chdir("E:/Data/Client1/Excel Files")

# Create a list with all the files
path = os.getcwd()
files = os.listdir(path)
```

I used list comprehension to get all the Excel workbooks. I could have used glob but wanted to use list comprehension for practice


```python
# Select only xlsx files
files_xlsx = [f for f in files if f[-4:] == "xlsx"]
```

I initialized an empty dataframe then imported the into the dataframe. I added the date and time as a field in this step, also replacing spaces and symbols in the column names. Then I exported each as a pipe delimited text file, with the filename being the date and time and the text files having no header or index.


```python
# Initialize empty dataframe
df = pd.DataFrame()

# Loop over list of Excel files, import into dataframe, add date field, and export
for f in files_xlsx:
    df = pd.read_excel(f, skiprows=4, skipfooter=3)
    # Add the date and time as a field
    df['file'] = f[-15:-5]
    # Replace spaces and symbols in column names
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df.columns = [c.replace('#', '') for c in df.columns]
    col_names = df.columns
    # Export each as a text file
    df.to_csv(f[-15:-5]+".txt", sep="|", index=None, header=None)
```

I used glob to put the names of all the text files in a list.


```python
# Get the .txt files
files_txt = glob.glob("*.txt")
```

I initialized a new dataframe to hold all the data as well as an empty list.


```python
# Initialize the dataframe to hold all the data
df_full = pd.DataFrame()

# Initialize an empty list
df_list = []
```

Then I imported the files from file_txt, appending the data to the list. Then I added the data in the list to the dataframe using the pandas concat function.


```python
# Loop over list of text files and import the data, then append the data to a list
for f in files_txt:
    data = pd.read_csv(f, sep="|", header=None)
    df_list.append(data)

# Add the data from the list to the dataframe for the full data set
df_full = pd.concat(df_list, axis=0)
```

Since column names aren't in the text files, I created a list that has them and then added those (the names have been changed except for file; that's the one I added based on the Excel filename).


```python
# Create the list with the column names
col_names_full = ['Field1', 'Field2', 'Field3', 'Field4', 'Field5',
                  'Field6', 'Field7', 'Field8', 'Field9', 'Field10',
                  'Field11', 'Field12', 'Field13', 'file']

# Add the column names to the dataframe
df_full.columns = col_names_full
```

Field4 has some NaN values that need to be replaced


```python
# Replace NaN in Field4 with "None"
df_full["Field4"] = df_full.Field4.fillna("None")
```

I created a dataframe that only contained a specific value from Field4, RSV.


```python
# Create RSV only dataframe
df_RSV = df_full[df_full.Field4 == "RSV"]
```

I created two outfile variables, one for the full dataframe and one for the RSV data only. Then I exported both as text delimited files.


```python
# Outfile for RSV only
outfile_RSV = "Data_RSV_20180312.txt"

# Outfile for all data
outfile_full = "Data_All_20180312.txt"

# Export RSV as pipe delimited text
df_RSV.to_csv(outfile_RSV, sep="|", index=None)

# Export full as pipe delimited text
df_full.to_csv(outfile_full, sep="|", index=None)
```

This produced two clean pipe delimited text files that could easily be imported elsewhere.
