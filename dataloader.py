import numpy as np
import pyodbc

import helper


# This function takes in a DataFrame and returns the same dataframe with any columns of data type timedelta64[ns] converted to seconds. 
# This is done using the select_dtypes method to select the relevant columns, 
# and then using the dt attribute of the pandas timedelta type to access the seconds component. 
# The new columns are then added to the dataframe using the assign method.
def timetoseconds(df):
    cols = df.select_dtypes('timedelta64[ns]')
    return (df.assign(**{col:df[col].dt.seconds 
               for col in cols}))



# This function takes in a DataFrame, a table name (tbl), 
# a flag to indicate whether the dataframe has an index (hasindex), 
# and a dictionary (custom) to specify custom column data types for the target SQL table. 
# The default value for hasindex is True, and the default value for custom is {"id":"INT PRIMARY KEY"}.
def full_load(df, tbl, hasindex = True, custom={"id":"INT PRIMARY KEY"}):
    # Here, if hasindex is True, the dataframe's index is reset using the reset_index method. 
    # The index is then set to be a sequence of integers, 
    # with the first value as 1 and the last value as the number of rows in the dataframe. 
    # Then, the reset_index method is used again, with the column name renamed to id using the rename method. 
    # The function then pipes the dataframe into the timetoseconds function to convert any timedelta columns to seconds.
    if hasindex:
        df = df.reset_index()
    df.index = id= np.arange(1,len(df)+1)
    df = (df.reset_index()
            .rename(columns = {'index': 'id'})
            #.assign(id = lambda x: x['id']+1)
            .pipe(timetoseconds)
            )
    # The dataframe is split into a list of smaller dataframes, 
    # each containing at most 100,000 rows, using the np.array_split function.
    list_df = np.array_split(df, len(df)//100000 +1)
    with pyodbc.connect(helper.get_connstring()) as conn:
        helper.to_sqlserver(df =list_df[0], name=tbl, conn = conn, if_exists="replace", custom=custom, temp=False)
        conn.commit()
        for i in range(1,len(list_df)):
            helper.to_sqlserver(df =list_df[i], name=tbl, conn = conn, if_exists="append", temp=False)
            conn.commit()
    conn.close()
    print(f'DataFrame full loaded to Table: {tbl}')
    