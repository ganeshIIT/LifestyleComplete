import pandas as pd
import numpy as np
import pyodbc
from functools import wraps
import re

from sqlalchemy import create_engine

import errors

# Global
DTYPE_MAP = {
    "int64": "int",
    "float64": "float",
    "object": "varchar(max)",
    "datetime64[ns]": "datetime2",
    "bool": "bit",
    "boolean": "bit",
    # To do - map timedelta64[ns] to seconds or string
}


def get_connstring(driver='{ODBC Driver 17 for SQL Server}'):
    """
    This function returns a connection string for SQL Server using pyodbc library. 
    
    The connection string consists of different parameters such as server name, database name, username, and password
    which are required to establish a connection with the SQL Server database.
    
    :param driver: A string representing the ODBC driver to use. The default value is '{ODBC Driver 17 for SQL Server}'
    
    :return: A string representing the connection string for the SQL Server database.
    
    usage:
    
    cstring = get_connstring()
    with pyodbc.connect() as conn:
        ## code
        
    """
    # Open the 'settings.txt' file in read mode
    with open('settings.txt', mode='r') as f:
        # Read the first line of the file and remove the newline character '\n'
        cs = f.readline().replace('\n', '')
    # Convert the string read from the file into a dictionary with the format {key:value}
    d = dict(x.split(':') for x in cs.split(' '))
    # Extract the values of the keys 'server', 'username', 'password', and 'db' from the dictionary
    server = d['server']
    username = d['username']
    password = d['password']
    database = d['db']
    # Create the connection string with the extracted values and the provided driver
    cs = 'DRIVER=' + driver + ';SERVER=tcp:' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password
    # Return the connection string
    return cs


def sqlalchemyengine(driverversion=17):
    """
    This function returns an SQLAlchemy engine for connecting to a database.

    The function uses the 'settings.txt' file to retrieve the database connection details. The file is read in 'r' mode and the first line of the file is retrieved and split using a comma separator. The resulting values are stored in a dictionary 'd'.
    The dictionary is then used to extract the server name, username, password, and database name which are used to form the engine string. The engine string is in the format "mssql+pyodbc://{username}:{password}@{server}:1433/{database}?driver=ODBC+Driver+{driverversion}+for+SQL+Server".
    The function returns the engine.
    """
    with open('settings.txt', mode='r') as f:
        cs = f.readline().replace('\n', '')
    d = dict(x.split(':') for x in cs.split(','))
    server = d['server']
    username = d['username']
    password = d['password'].replace('@', '%40')
    database = d['db']
    engine = create_engine(
        f"mssql+pyodbc://{username}:{password}@{server}:1433/{database}?driver=ODBC+Driver+{driverversion}+for+SQL+Server"
    )
    return engine


def _cleanlabel(label):
    """
    This function takes a label as input and returns a cleaned version of the label.
    The function uses the `re` module to remove all non-alphanumeric and non-whitespace characters from the label. It then converts the label to lowercase and replaces any spaces with underscores.
    """
    return re.sub(r'[^\w\s]', '', str(label)).lower().replace(' ', '_')


def cleanlabel(method):
    """
    This is a decorator function for cleaning the column labels and index label of a DataFrame.

    The function takes a method as input and returns a wrapped method. The wrapped method is used to return a DataFrame. The function uses the `wraps` method from the `functools` module to preserve the metadata of the method being decorated.
    The function cleans the column labels and the index label of the DataFrame returned by the decorated method. The cleaning process is performed using the `_cleanlabel` function which removes all non-alphanumeric and non-whitespace characters from the label, converts it to lowercase and replaces any spaces with underscores.
    """

    @wraps(method)
    def method_wrapper(*args, **kwargs):
        df = method(*args, **kwargs)
        df.columns = [_cleanlabel(col) for col in df.columns]
        if df.index.name:
            df.index.rename(_cleanlabel(df.index.name), inplace=True)
        return df

    return method_wrapper


# def int64_to_uint8(df):
#     cols = df.select_dtypes('int64')
#     return (df.astype({col: 'uint8' for col in cols}))

# def datetime64_to_date(df):
#     cols = df.select_dtypes('datetime64[ns]').columns
#     for col in cols:
#         df[col] = df[col].dt.normalize()
#     return df

# def flatten_cols(df):
#     cols = ['_'.join(map(str, vals)) for vals in df.columns.to_flat_index()]
#     df.columns = cols
#     return df

# def get_row_count(*dfs):
#     return [df.shape[0] for df in dfs]

# def timetoseconds(df):
#     """ Converts timedelta to seconds to push to the db tables"""
#     cols = df.select_dtypes('timedelta64[ns]')
#     return (df.assign(**{col: df[col].dt.seconds for col in cols}))


def int64_to_uint8(df):
    """
    Convert the int64 dtype columns of a dataframe to uint8
    
    Parameters:
    df (pandas.DataFrame): Input dataframe

    Returns:
    pandas.DataFrame: Dataframe with int64 dtype columns converted to uint8
    """
    # get all the int64 columns in the dataframe
    cols = df.select_dtypes('int64')
    # convert the int64 columns to uint8
    return (df.astype({col: 'uint8' for col in cols}))


def datetime64_to_date(df):
    """
    Convert the datetime64[ns] dtype columns to date only
    
    Parameters:
    df (pandas.DataFrame): Input dataframe

    Returns:
    pandas.DataFrame: Dataframe with datetime64[ns] dtype columns converted to date only
    """
    # get the datetime64[ns] dtype columns
    cols = df.select_dtypes('datetime64[ns]').columns
    # convert the datetime64[ns] columns to date only
    for col in cols:
        df[col] = df[col].dt.normalize()
    return df


def flatten_cols(df):
    """
    Flatten the multi-level column labels of a dataframe
    
    Parameters:
    df (pandas.DataFrame): Input dataframe

    Returns:
    pandas.DataFrame: Dataframe with flattened column labels
    """
    # flatten the multi-level column labels
    cols = ['_'.join(map(str, vals)) for vals in df.columns.to_flat_index()]
    df.columns = cols
    return df


def get_row_count(*dfs):
    """
    Get the number of rows for multiple dataframes
    
    Parameters:
    *dfs (pandas.DataFrame): Input dataframes

    Returns:
    list: List of number of rows for each input dataframe
    """
    # return a list of the number of rows for each dataframe
    return [df.shape[0] for df in dfs]


def timetoseconds(df):
    """ 
    Convert timedelta to seconds for a dataframe
    
    Parameters:
    df (pandas.DataFrame): Input dataframe

    Returns:
    pandas.DataFrame: Dataframe with timedelta converted to seconds
    """
    # get the timedelta64[ns] dtype columns
    cols = df.select_dtypes('timedelta64[ns]')
    # convert the timedelta to seconds
    return (df.assign(**{col: df[col].dt.seconds for col in cols}))


# def validate_df(columns, instance_method=True):
#     """
#     Doc here
#     """

#     def method_wrapper(method):

#         @wraps(method)
#         def validate_wrapper(self, *args, **kwargs):
#             # functions and static methods don't pass self
#             # so self is the first positional argument in that case
#             df = (self, *args)[0 if not instance_method else 1]
#             if not isinstance(df, pd.DataFrame):
#                 raise ValueError("You should pass a pandas DataFrame")
#             if columns.difference(df.columns):
#                 raise ValueError(
#                     f'DataFrame must contain the following columns: {columns}')
#             return method(self, *args, **kwargs)

#         return validate_wrapper

#     return method_wrapper

# def window_calc(df, func, agg_dict, *args, **kwargs):
#     """
#         Perform window calculations
#     """
#     return df.pipe(func, *args, **kwargs).agg(agg_dict)


def validate_df(columns, instance_method=True):
    """
    Validate a passed dataframe and ensure it meets certain column requirements.

    Parameters:
    columns (set): The required set of columns that the dataframe must contain.
    instance_method (bool): Flag indicating whether the method using validate_df is an instance method or not. Defaults to True.

    Returns:
    method_wrapper: A decorator function that validates the dataframe and passes it to the decorated method.
    """

    def method_wrapper(method):
        """
        Decorator function that validates the dataframe and passes it to the decorated method.

        Parameters:
        method: The method to be decorated.

        Returns:
        validate_wrapper: A wrapper function that performs the validation and then calls the decorated method.
        """

        @wraps(method)
        def validate_wrapper(self, *args, **kwargs):
            # functions and static methods don't pass self
            # so self is the first positional argument in that case
            df = (self, *args)[0 if not instance_method else 1]
            if not isinstance(df, pd.DataFrame):
                raise ValueError("You should pass a pandas DataFrame")
            if columns.difference(df.columns):
                raise ValueError(
                    f'DataFrame must contain the following columns: {columns}')
            return method(self, *args, **kwargs)

        return validate_wrapper

    return method_wrapper


def window_calc(df, func, agg_dict, *args, **kwargs):
    """
    Perform window calculations on a pandas DataFrame.

    Parameters:
    df (pandas DataFrame): The DataFrame to perform window calculations on.
    func (function): The window function to apply to the DataFrame.
    agg_dict (dict): A dictionary mapping column names to aggregation functions to be applied to the result of the window function.
    *args: Additional positional arguments to pass to the window function.
    **kwargs: Additional keyword arguments to pass to the window function.

    Returns:
    pandas DataFrame: The result of the window calculations.
    """
    return df.pipe(func, *args, **kwargs).agg(agg_dict)


# def _check_duplicate_cols(df):
#     """Returns duplicate column names (case insensitive)
#     """
#     cols = [c.lower() for c in df.columns]
#     dups = [x for x in cols if cols.count(x) > 1]
#     if dups:
#         raise errors.DuplicateColumns(
#             f"There are duplicate column names. Repeated names are: {dups}. SQL Server dialect requires unique names (case insensitive)."
#         )

# def _clean_col_name(column):
#     """Removes special characters from column names
#     """
#     column = str(column).replace(" ", "_").replace("(", "").replace(
#         ")", "").replace("[", "").replace("]", "")
#     column = f"[{column}]"
#     return column

# def _clean_custom(df, custom):
#     """Validate and clean custom columns
#     """
#     for k in list(custom):
#         clean_col = _clean_col_name(k)
#         if clean_col not in df.columns:
#             raise errors.CustomColumnException(
#                 f"Custom column {k} is not in the dataframe.")
#         custom[clean_col] = custom.pop(k)
#     return custom


def _check_duplicate_cols(df):
    """
    Check if there are duplicate column names (case insensitive) in the dataframe.

    Args:
    df: pandas DataFrame
    
    Returns:
    Raises an error of type "DuplicateColumns" if there are duplicate column names.
    The error message includes the duplicate column names.
    """
    # Convert all column names to lower case
    cols = [c.lower() for c in df.columns]
    # Find column names that are repeated more than once
    dups = [x for x in cols if cols.count(x) > 1]
    # If there are duplicates, raise an error
    if dups:
        raise errors.DuplicateColumns(
            f"There are duplicate column names. Repeated names are: {dups}. SQL Server dialect requires unique names (case insensitive)."
        )


def _clean_col_name(column):
    """
    Remove special characters from a column name.

    Args:
    column: column name (string)

    Returns:
    Cleaned column name surrounded by square brackets.
    """
    # Replace spaces, parentheses, and square brackets with underscores
    column = str(column).replace(" ", "_").replace("(", "").replace(
        ")", "").replace("[", "").replace("]", "")
    # Add square brackets around the cleaned column name
    column = f"[{column}]"
    return column


def _clean_custom(df, custom):
    """
    Validate and clean custom columns

    Args:
    df: pandas DataFrame
    custom: dictionary of custom columns with key as the original column name and value as the custom value

    Returns:
    Cleaned custom columns as a dictionary with key as the cleaned column name and value as the custom value.
    Raises an error of type "CustomColumnException" if a custom column is not in the dataframe.
    """
    # Iterate over the keys in the custom dictionary
    for k in list(custom):
        # Clean the column name
        clean_col = _clean_col_name(k)
        # If the cleaned column name is not in the dataframe, raise an error
        if clean_col not in df.columns:
            raise errors.CustomColumnException(
                f"Custom column {k} is not in the dataframe.")
        # Replace the original column name with the cleaned column name in the custom dictionary
        custom[clean_col] = custom.pop(k)
    return custom


# def _get_data_types(df, custom):
#     """Get data types for each column as dictionary
#     Handles default data type assignment and custom data types
#     """
#     data_types = {}
#     for c in list(df.columns):
#         if c in custom:
#             data_types[c] = custom[c]
#             continue
#         dtype = str(df[c].dtype)
#         if dtype not in DTYPE_MAP:
#             data_types[c] = "varchar(255)"
#         else:
#             data_types[c] = DTYPE_MAP[dtype]
#     return data_types

# def _get_default_schema(cur: pyodbc.Cursor) -> str:
#     """Get the default schema of the caller
#     """
#     return str(cur.execute("select SCHEMA_NAME() as scm").fetchall()[0][0])

# def _get_schema(cur: pyodbc.Cursor, table_name: str):
#     """Get schema and table name - returned as tuple
#     """
#     t_spl = table_name.split(".")
#     if len(t_spl) > 1:
#         return t_spl[0], ".".join(t_spl[1:])
#     else:
#         return _get_default_schema(cur), table_name

# def _clean_table_name(table_name):
#     """Cleans the table name
#     """
#     return table_name.replace("'", "''")


# Function to retrieve data types of each column in the dataframe
def _get_data_types(df, custom):
    """
    Get data types for each column as dictionary
    Handles default data type assignment and custom data types
    """
    data_types = {}  # create an empty dictionary to store the data types
    # loop through the columns in the dataframe
    for c in list(df.columns):
        if c in custom:
            # if the current column is in the custom dictionary, add the value of that column in custom to the data_types dictionary
            data_types[c] = custom[c]
            continue
        dtype = str(df[c].dtype)  # get the data type of the current column
        if dtype not in DTYPE_MAP:
            # if the data type is not in the DTYPE_MAP, set the value to "varchar(255)"
            data_types[c] = "varchar(255)"
        else:
            # else, set the value to the data type from DTYPE_MAP
            data_types[c] = DTYPE_MAP[dtype]
    return data_types


# Function to retrieve the default schema of the caller
def _get_default_schema(cur: pyodbc.Cursor) -> str:
    """
    Get the default schema of the caller
    """
    # return the schema name using the SQL query and fetching the first element of the first row of the result
    return str(cur.execute("select SCHEMA_NAME() as scm").fetchall()[0][0])


# Function to retrieve schema and table name from the table_name argument


def _get_schema(cur: pyodbc.Cursor, table_name: str):
    """
    Get schema and table name - returned as tuple
    """
    t_spl = table_name.split(".")  # split the table_name based on the "."
    if len(t_spl) > 1:
        # if there is more than one item after splitting, return the first item as schema and the rest as table
        return t_spl[0], ".".join(t_spl[1:])
    else:
        # else, return the default schema and the table_name
        return _get_default_schema(cur), table_name


# Function to clean the table name
def _clean_table_name(table_name):
    """
    Cleans the table name
    """
    # replace single quotes with two single quotes to escape them
    return table_name.replace("'", "''")


# def _check_exists(cur, schema, table, temp):
#     """Check in conn if table exists
#     """
#     if temp:
#         return cur.execute(
#             f"IF OBJECT_ID('tempdb..#[{table}]') IS NOT NULL select 1 else select 0"
#         ).fetchall()[0][0]
#     else:
#         return cur.execute(
#             f"IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}' and TABLE_SCHEMA = '{schema}') select 1 else select 0"
#         ).fetchall()[0][0]

# def _generate_create_statement(schema, table, cols, temp):
#     """Generates a create statement
#     """
#     cols = ",".join([f'\n\t{k} {v}' for k, v in cols.items()])
#     schema_if_temp = f"[#{table}]" if temp else f"[{schema}].[{table}]"
#     return f"create table {schema_if_temp}\n({cols}\n)"

# def _check_parameter_if_exists(if_exists):
#     """Raises an error if parameter 'if_exists' is not correct
#     """
#     if if_exists not in ('append', 'fail', 'replace'):
#         raise errors.WrongParam(
#             f"Incorrect parameter value {if_exists} for 'if_exists'. Can be 'append', 'fail', or 'replace'"
#         )


def _check_exists(cur, schema, table, temp):
    """
    Check if the table exists in the database by using the passed `cur` connection.
    
    Parameters:
    - cur: pyodbc.Cursor object
      Cursor to the database to run the check against
    - schema: str
      The name of the schema of the table
    - table: str
      The name of the table
    - temp: bool
      If the table is temporary, `temp` will be True
      
    Returns:
    int
      1 if table exists, 0 if not
    """
    if temp:
        # Check if the table exists in the tempdb
        return cur.execute(
            f"IF OBJECT_ID('tempdb..#[{table}]') IS NOT NULL select 1 else select 0"
        ).fetchall()[0][0]
    else:
        # Check if the table exists in the specified schema
        return cur.execute(
            f"IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}' and TABLE_SCHEMA = '{schema}') select 1 else select 0"
        ).fetchall()[0][0]


def _generate_create_statement(schema, table, cols, temp):
    """
    Generates a CREATE TABLE statement to create the table in the database.
    
    Parameters:
    - schema: str
      The name of the schema of the table
    - table: str
      The name of the table
    - cols: dict
      A dictionary of the columns and their data types to create in the table
    - temp: bool
      If the table is temporary, `temp` will be True
      
    Returns:
    str
      The CREATE TABLE statement to create the table in the database
    """
    # Join the dictionary of columns and data types into a string for the CREATE TABLE statement
    cols = ",".join([f'\n\t{k} {v}' for k, v in cols.items()])
    schema_if_temp = f"[#{table}]" if temp else f"[{schema}].[{table}]"
    return f"create table {schema_if_temp}\n({cols}\n)"


def _check_parameter_if_exists(if_exists):
    """
    Raises an error if the `if_exists` parameter is not one of the accepted values.
    
    Parameters:
    - if_exists: str
      The value of the `if_exists` parameter
    
    Raises:
    errors.WrongParam: If `if_exists` is not one of the accepted values ('append', 'fail', or 'replace')
    """
    if if_exists not in ('append', 'fail', 'replace'):
        raise errors.WrongParam(
            f"Incorrect parameter value {if_exists} for 'if_exists'. Can be 'append', 'fail', or 'replace'"
        )


# This function takes a pandas DataFrame and writes it to a SQL server database using the PyODBC library. The function takes in several parameters:

# df: The pandas DataFrame to upload to SQL server
# name: A string representing the desired name of the table in SQL server
# conn: A valid PyODBC connection object
# if_exists: A string option that determines what to do if the specified table name already exists in the database. The options are 'append', 'fail', and 'replace'. If the table does not exist, a new one will be created. By default, this option is set to 'append'
# custom: A dictionary object with one or more of the column names being uploaded as the key, and a valid SQL column definition as the value. The value must contain a type (INT, FLOAT, VARCHAR(500), etc.), and can optionally also include constraints (NOT NULL, PRIMARY KEY, etc.)
# temp: A boolean indicating whether to create a local SQL server temporary table for the connection or not. By default, it is set to False
# copy: A boolean indicating whether to make a copy of the dataframe or not. By default, it is set to False

# The function performs several operations to prepare the data for insertion into SQL server:
# - Cleaning the table name and column names to be used as valid SQL names
# - Checking for duplicate column names and cleaning the custom dictionary
# - Assigning data types to the columns
# - Checking if the table exists in SQL server, handling the table based on the value of the if_exists parameter
# - Generating the SQL statement to create the table if it doesn't exist
# - Inserting the data into the table using the PyODBC library's fast_executemany method

# The function returns the create statement that was used to create the table in SQL server.


def to_sqlserver(df,
                 name,
                 conn,
                 if_exists='append',
                 custom=None,
                 temp=False,
                 copy=False):
    """Main fast_to_sql function.
    Writes pandas dataframe to sql server using pyodbc fast_executemany
    
    
    df: pandas DataFrame to upload
    name: String of desired name for the table in SQL server
    conn: A valid pyodbc connection object
    if_exists: Option for what to do if the specified table name already exists in the database. If the table does not exist a new one will be created. By default this option is set to 'append'
    'append': Appends the dataframe to the table if it already exists in SQL server.
    'fail': Purposely raises a FailError if the table already exists in SQL server.
    'replace': Drops the old table with the specified name, and creates a new one. Be careful with this option, it will completely delete a table with the specified name in SQL server.
    custom: A dictionary object with one or more of the column names being uploaded as the key, and a valid SQL column definition as the value. The value must contain a type (INT, FLOAT, VARCHAR(500), etc.), and can optionally also include constraints (NOT NULL, PRIMARY KEY, etc.)
    Examples: {'ColumnName':'varchar(1000)'} {'ColumnName2':'int primary key'}
    temp: Either True if creating a local sql server temporary table for the connection, or False (default) if not.
    copy: Defaults to False. If set to True, a copy of the dataframe will be made so column names of the original dataframe are not altered.
    """

    if copy:
        df = df.copy()

    # Assign null custom
    if custom is None:
        custom = {}

    # Handle series
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Clean table name
    name = _clean_table_name(name)

    # Clean columns
    columns = [_clean_col_name(c) for c in list(df.columns)]
    df.columns = columns

    # Check for duplicate column names
    _check_duplicate_cols(df)
    custom = _clean_custom(df, custom)

    # Assign data types
    data_types = _get_data_types(df, custom)

    # Get schema
    cur = conn.cursor()
    schema, name = _get_schema(cur, name)
    if schema == '':
        schema = cur.execute("SELECT SCHEMA_NAME()").fetchall()[0][0]
    exists = _check_exists(cur, schema, name, temp)

    # Handle existing table
    create_statement = ''
    if exists:
        _check_parameter_if_exists(if_exists)
        if if_exists == "replace":
            cur.execute(f"drop table [{schema}].[{name}]")
            create_statement = _generate_create_statement(
                schema, name, data_types, temp)
            cur.execute(create_statement)
        elif if_exists == "fail":
            fail_msg = f"Table [{schema}].[{name}] already exists." if temp else f"Temp table #[{name}] already exists in this connection"
            raise errors.FailError(fail_msg)
    else:
        create_statement = _generate_create_statement(schema, name, data_types,
                                                      temp)
        cur.execute(create_statement)

    # Run insert
    if temp:
        insert_sql = f"insert into [#{name}] values ({','.join(['?' for v in data_types])})"
    else:
        insert_sql = f"insert into [{schema}].[{name}] values ({','.join(['?' for v in data_types])})"
    insert_cols = df.values.tolist()
    insert_cols = [[
        None if type(cell) == float and np.isnan(cell) else cell
        for cell in row
    ] for row in insert_cols]
    cur.fast_executemany = True
    cur.executemany(insert_sql, insert_cols)
    cur.close()
    return create_statement


if __name__ == "__main__":
    # using connection function
    cstring = get_connstring()
    with pyodbc.connect(cstring) as conn:
        query = "SELECT top 10 * from listings"
        df = pd.read_sql(query, conn)
        print(df)

    ## using sqlalchemy
    # engine = sqlalchemyengine()
    # connection = engine.connect()
    # query = "SELECT top 10 * from health"
    # df = pd.read_sql(query, connection)
    # print(df)
    # connection.close()
