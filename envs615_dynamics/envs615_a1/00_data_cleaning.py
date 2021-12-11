from datetime import date
import pandas as pd
from googletrans import Translator


def read_clean_data(file: str):
    """Read in and clean the Kaggle car data.

    Clean by removing outlier values from the various variables.
    These values were identified prior to this analysis. Also
    removes large number of columns that don't provide any useful
    information.
    Provides outputs, the clean file, and a boolean series giving integer
    postions of dropped columns

    Args:
        file (str): the file and path location to read in
    """
    # read in the data, special encoding required
    df = pd.read_csv(file, encoding="latin-1")
    # remove some questionable values, due to large number of rows
    # consider it fine to remove values that are possibly real but
    # will not improve models
    invalid_year = (df['yearOfRegistration'] > 2020) |\
                   (df['yearOfRegistration'] < 1950)
    invalid_price = (df['price'] > 100000) |\
                    (df['price'] < 100)
    invalid_power = (df['powerPS'] > 500) |\
                    (df['powerPS'] < 50)
    # variable used to remove outliers
    match = invalid_price | invalid_year | invalid_power
    match_dropped = {"Number of dropped rows:": match.sum(),
                     "Percentage of dropped rows:": (match.sum()/len(df))}
    # drop matched rows, also remove columns that are not required
    df = df[-match]\
        .drop([
            'dateCrawled',
            'seller',  # almost all one value
            'offerType',  # almost all one value
            'name',  # too many unique values
            'abtest',  # uninterpretable ebay code
            'lastSeen',
            'nrOfPictures',  # all zero
            'dateCreated',
            'model'  # far too many values
        ], axis=1)
    return df, match_dropped


df, match_dropped = read_clean_data("./data/autos.csv")
match_dropped  # count of rows removed in cleaning
# find number of rows with NA values
print("Total NA Values:", df.shape[0] - df.dropna().shape[0])
df = df.dropna()  # drop NA rows


def translate_rows_de(df: pd.DataFrame, cols: list):
    """Translate selected rows in a dataframe from de to en.

    Utilises googles translation API to accurately translate text
    contained within a dataframe. For efficient use first the number of
    unique values in each chosen column are found and translated. These
    may then be added back into the dataframe using pandas.replace.
    This avoids applying the translation API to each row, which does not
    work with a large dataframe.

    Args:
        df (pd.DataFrame): a dataframe with values that require translation
        cols (list): columns containing rows to translate
    """
    # initialise google Translator object
    translator = Translator()
    translations = {}  # empty dict to store translations
    # loop through each column specified above in the df
    # find unique categorical values for each
    # for each unique value per row translate from de to en
    # store the translation in the dictionary
    for column in cols:
        # unique elements of the column
        unique_elements = df[column].unique()
        for element in unique_elements:
            # add translation to the dictionary
            translations[element] = translator.translate(
                element, src='de', dest='en').text
    df = df.replace(translations)
    return df


# columns that require translation from german
cols = [
    'vehicleType',
    'notRepairedDamage',
    'fuelType',
    'gearbox'
]

# run translation function on predetermined rows
df = translate_rows_de(df, cols=cols)

# one translation needs correcting
df = df\
    .replace('manually', 'manual')
df = df.rename(columns={'notRepairedDamage': 'damaged'})  # rename for clarity


def find_age(df: pd.DataFrame, year_col: str, month_col: str):
    """Convert the dataframe to work with date manipulation.

    Both *_col inputs must evaluate to a column found within the
    dataframe, with a year and month in the correct format.

    Args:
        df (pd.DataFrame): [TODO:description]
        year_col (str): year in the format: 2020
        month_col (str): month in the format: 01
    """
    # create date type from month and year of registration
    date_reg = df.rename(
        columns={
            year_col: 'year',
            month_col: 'month'
        }
    )[['year', 'month']]
    date_reg['day'] = '1'  # required for to_datetime
    # change to date class, coerce removes invalid dates
    date_reg = pd.to_datetime(date_reg, errors='coerce')
    # subtract date from current date to find age

    # run function days_subtract on all rows
    date_reg = date_reg.apply(lambda x: pd.to_datetime(date.today()) - x)
    df['age'] = date_reg.dt.days/365  # age in years
    df = df.drop([year_col, month_col], axis=1)  # remove old cols
    return df


df = find_age(
    df,
    year_col='yearOfRegistration',
    month_col='monthOfRegistration'
)

# invalid dates are now classified as NA, remove them
# find number of rows with NA values
print("Number of NA Values:", df.shape[0] - df.dropna().shape[0])
df = df.dropna()  # drop NA rows


def join_postcodes(df: pd.DataFrame, pc: str):
    # read in detailed postcode data from GeoNames
    # keep only largest authority district
    # remove duplicate rows (since smaller authorities are gone)
    pc_df = pd.read_csv(pc, sep="\t", header=None)\
        .iloc[:, [1, 3]]\
        .drop_duplicates()
    # rename for join
    pc_df.columns = ['postalCode', 'area']
    # setup a join to include new area column with associated postcode
    df = df.set_index('postalCode')\
        .join(pc_df.set_index('postalCode'))\
        .reset_index()
    return df


# DE postcodes from GeoNames
df = join_postcodes(df, "./data/DE.txt")

# improved category, far fewer unique values
# number of unique postcodes
len(df['postalCode'].unique())
# number of unique areas
len(df['area'].unique())
df = df.drop('postalCode', axis=1)

df.to_csv("./data/derived/cars_cleaned.csv")
