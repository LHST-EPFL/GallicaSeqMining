# %%
import multiprocessing
from functools import lru_cache
import re
import pandas as pd
from tqdm import tqdm
from dataprep.eda import create_report
from fastuaparser import parse_ua
import dask.dataframe as dd
from dask import bag as db
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler
from dask.distributed import Client
from fastuaparser import parse_ua
import os
# deactivating warnings
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
# show full length of strings
pd.set_option('display.max_colwidth', -1)

# %%
# parameters
PATTERN = r'^\[(?P<timestamp>.*?)\]\s+"(?P<request_type>\w+)\s+(?P<endpoint>.*?)\s+(?P<http_version>HTTP/\d\.\d)+"\s+(?P<status_code>\d+)?\s*(?P<content_length>\d+|\-)?\s+"(?P<referrer>.*?)"\s+"(?P<user_agent>.*?)"?\"?\s*(?P<response_time>\d+)?$'
# number of files to be processed (should be between 1 and 22637)
TOTAL_NUMBER_OF_FILES = 6575
CHUNK_SIZE = 20

if TOTAL_NUMBER_OF_FILES == 'max':
    TOTAL_NUMBER_OF_FILES = 6575

# %% [markdown]
# ## I. Reading the dataframe

files_start_indices = list(range(0, TOTAL_NUMBER_OF_FILES, CHUNK_SIZE))
files_stop_indices = list(range(CHUNK_SIZE, TOTAL_NUMBER_OF_FILES, CHUNK_SIZE))
files_stop_indices.append(TOTAL_NUMBER_OF_FILES - 1)

# if the path "temp_data/processed_parquet" does not exist, create it
if not os.path.exists("temp_data/processed_parquet"):
    os.makedirs("temp_data/processed_parquet")
    
def process_chunk(start_index, stop_index, file_number):
    # read the file paths from csv (the first line is the header)
    df_files = pd.read_csv("temp_data/files.csv")
    # restrict the number of files to be processed 
    file_paths = df_files.iloc[start_index:stop_index]['local_path'].tolist()
    # define the columns of the dataframe
    cols = ["hash", "user", "country", "city", "request"]
    # column dtypes
    column_dtypes = {
        'hash': 'string',
        'user': 'string',
        'country': 'string',
        'city': 'string',
        'request': 'string'
    }

    # Function to read a single file and return a Dask DataFrame


    def read_file(file_path):
        ddf = dd.read_csv(file_path, sep="##", header=None,
                        names=cols, compression="gzip", dtype=column_dtypes)
        return ddf


    # Read files one by one and update the progress bar
    dfs = []
    with tqdm(total=len(file_paths), desc="Processing files", unit="file") as progress_bar:
        for file_path in file_paths:
            df = read_file(file_path)
            dfs.append(df)
            progress_bar.update(1)

    # Concatenate all Dask DataFrames
    with ProgressBar():
        ddf = dd.concat(dfs, axis=0, ignore_index=True)



    # ## II. Cleaning and Extracting features


    # get the request to the proper format (strip any leading spaces or characters before the first bracket)
    ddf["request"] = ddf["request"].str.lstrip(" -")


    # Extract the different components of the log request using vectorized string operations and regular expressions including a flag when the extraction fails
    ddf_log = ddf["request"].str.extract(
        PATTERN,
        expand=True,
        flags=re.IGNORECASE
    )

    # Concatenate the two dataframes
    ddf = dd.concat([ddf, ddf_log], axis=1)
    # drop na from timestamp
    ddf = ddf.dropna(subset=['timestamp'])


    # the string'-' in the referrer and user_agent columns should be converted to NaN
    ddf["referrer"] = ddf["referrer"].replace("-", pd.NA)
    ddf["user_agent"] = ddf["user_agent"].replace("-", pd.NA)
    # drop unused columns
    # List the columns you want to drop
    columns_to_drop = ['hash', 'request', 'http_version',
                    'content_length', 'response_time']
    # Drop the specified columns
    ddf = ddf.drop(columns_to_drop, axis=1)
    # extract the ark from the request type column
    # df['ark'] = df['endpoint'].str.extract(r'12148/([a-zA-Z0-9]+)')
    ddf_ark = ddf["endpoint"].str.extract(
        r"12148/(?P<Ark>[a-zA-Z0-9]+)", expand=True)
    # concatenate the two dataframes
    ddf = dd.concat([ddf, ddf_ark], axis=1)
    # drop duplicates on timestamp and endpoint
    ddf = ddf.drop_duplicates(subset=["timestamp", "endpoint"])


    # Remove leading slashes from the 'endpoint' column
    ddf['endpoint'] = ddf['endpoint'].str.lstrip('/')

    # Split the 'endpoint' column once and store the result in a temporary variable
    split_endpoint = ddf['endpoint'].str.split('/', expand=True, n=4)

    # Check first characters until the first slash in the 'endpoint' column
    ddf['endpoint_1'] = split_endpoint[0].fillna('')

    # Check the characters between the first slash and second slash in the 'endpoint' column
    ddf['endpoint_2'] = split_endpoint[1].fillna('')

    # Check the characters between the second slash and third slash in the 'endpoint' column
    ddf['endpoint_3'] = split_endpoint[2].fillna('')

    # Check the characters between the third slash and fourth slash in the 'endpoint' column
    ddf['endpoint_4'] = split_endpoint[3].fillna('')

    # concatenate the first 4 endpoint columns into column endpoint radical
    ddf['endpoint_radical'] = (ddf['endpoint_1'] + '/' + ddf['endpoint_2'] +
                            '/' + ddf['endpoint_3'] + '/' + ddf['endpoint_4']).replace('///', '')
    # if 'ark' is present in the endpoint_radical, remove all characters after the slash
    ddf['endpoint_radical'] = ddf['endpoint_radical'].str.replace('ark:.*', 'ark:')


    # ## Request Classification


    # if the endpoint is an empty string it is homepage
    ddf['is_homepage'] = 0
    ddf['is_homepage'] = ddf['is_homepage'].mask(ddf['endpoint'] == '', 1)
    # if download is present in the endpoint and ark is not nan, set request_class to download
    ddf['is_download'] = 0
    ddf['is_download'] = ddf['is_download'].mask((ddf['endpoint'].str.contains('download') & (
        ddf['endpoint_1'] == 'ark:')) | ddf['endpoint'].str.contains('services/ajax/action/download/'), 1)
    # is document download if is download and enpoint contains the string 'services/ajax/action/download'
    ddf["is_document_download"] = 0
    ddf["is_document_download"] = ddf["is_document_download"].mask(
        (ddf["is_download"] == 1) & (ddf["endpoint"].str.contains("services/ajax/action/download")), 1)
    # is page download if is download and enpoint ends with the string 'download=1'
    ddf["is_page_download"] = 0
    ddf["is_page_download"] = ddf["is_page_download"].mask(
        (ddf["is_download"] == 1) & (ddf["endpoint"].str.endswith("download=1")), 1)
    # if ark is present in the endpoint_1 set is_document to 1
    ddf['is_document'] = 0
    ddf['is_document'] = ddf['is_document'].mask(
        (ddf['endpoint_1'] == 'ark:') & (ddf['is_download'] != 1), 1)
    # if iiif is present in the endpoint_1 set is_iiif to 1
    ddf['is_iiif'] = 0
    ddf['is_iiif'] = ddf['is_iiif'].mask(ddf['endpoint_1'] == 'iiif', 1)
    # if assets or html is present in the endpoint_1 and endpoint 2 is not und , set request_class to static
    ddf['is_static_http'] = 0
    ddf['is_static_http'] = ddf['is_static_http'].mask(((ddf['endpoint_1'] == 'assets') | (
        ddf['endpoint_1'] == 'html')) & (ddf['endpoint_2'] != 'und'), 1)
    # if blog is present in the endpoint_1 , set request_class to blog
    ddf['is_blog'] = 0
    ddf['is_blog'] = ddf['is_blog'].mask(ddf['endpoint_1'] == 'blog', 1)
    # if services is present in the endpoint_1 , set request_class to services
    ddf['is_services'] = 0
    endpoint_3_exceptions = ['search', 'pagination', 'action', 'mode']
    ddf['is_services'] = ddf['is_services'].mask((ddf['endpoint_1'] == 'services') & (
        ~ddf['endpoint_3'].isin(endpoint_3_exceptions)), 1)
    # if search is present in the endpoint_radical , set request_class to search
    ddf['is_search'] = 0
    ddf['is_search'] = ddf['is_search'].mask(
        ddf['endpoint_radical'].str.contains('services/engine/search'), 1)
    # is simple search if is search and enpoint does not contain the string 'advancedSearch' nor 'subsearch' nor 'restrictedSearch'
    ddf["is_simple_search"] = 0
    ddf["is_simple_search"] = ddf["is_simple_search"].mask(
        (ddf["is_search"] == 1) & (~ddf["endpoint"].str.contains("advancedSearch|subsearch|restrictedSearch")), 1)
    # is advanced search if is search and enpoint contains the string 'advancedSearch'
    ddf["is_advanced_search"] = 0
    ddf["is_advanced_search"] = ddf["is_advanced_search"].mask(
        (ddf["is_search"] == 1) & (ddf["endpoint"].str.contains("advancedSearch")), 1)
    # is filtering search results if is search and enpoint contains the string 'subsearch' or 'restrictedSearch'
    ddf["is_filtering_search_results"] = 0
    ddf["is_filtering_search_results"] = ddf["is_filtering_search_results"].mask(
        (ddf["is_search"] == 1) & (ddf["endpoint"].str.contains("subsearch|restrictedSearch")), 1)
    # if pagination is present in the endpoint_radical , set request_class to search
    ddf['is_pagination'] = 0
    ddf['is_pagination'] = ddf['is_pagination'].mask(
        ddf['endpoint'].str.contains('services/ajax/pagination'), 1)
    # is mode if enpoint contains the string 'services/ajax/mode' and does not contain the string 'zoom'
    ddf['is_mode'] = 0
    ddf['is_mode'] = ddf['is_mode'].mask(
        ddf['endpoint_radical'].str.contains('services/ajax/mode') & (
            ~ddf['endpoint'].str.contains('zoom')), 1)
    # is zoom if enpoint contains the string 'services/ajax/mode' and contains the string 'zoom'
    ddf['is_zoom'] = 0
    ddf['is_zoom'] = ddf['is_zoom'].mask(
        ddf['endpoint_radical'].str.contains('services/ajax/mode') & (
            ddf['endpoint'].str.contains('zoom')), 1)
    # if endpoint 1 is html and and endpoint 2 is und , set request_class to heading
    ddf['is_heading'] = 0
    ddf['is_heading'] = ddf['is_heading'].mask(
        (ddf['endpoint_1'] == 'html') & (ddf['endpoint_2'] == 'und'), 1)

    # Create a new column with the sum of all "is_" columns for each row
    ddf['is_sum'] = ddf[['is_homepage', 'is_document', 'is_iiif', 'is_static_http', 'is_blog', 'is_zoom',
                        'is_services', 'is_simple_search', 'is_advanced_search', 'is_filtering_search_results', 'is_pagination', 'is_heading', 'is_page_download', 'is_document_download', 'is_mode']].sum(axis=1)
    actions = ['is_homepage', 'is_document', 'is_blog', 'is_simple_search', 'is_advanced_search', 'is_filtering_search_results', 'is_page_download', 'is_document_download',
            'is_pagination', 'is_heading', 'is_mode', 'is_zoom']
    # create new dataframe based on usefulness
    ddf_all = ddf.copy()
    # ddf_useful where column of actions are 1
    ddf = ddf.loc[(ddf[actions] == 1).any(axis=1)]
    ddf_plus = ddf_all.loc[(ddf_all["is_sum"] > 1)]
    ddf_0 = ddf_all.loc[(ddf_all["is_sum"] == 0)]

    ddf_plus["cooccurent"] = ddf_plus[actions].apply(
        lambda x: ",".join(x[x == 1].index), axis=1)


    # ### Document requests parsing


    # Pre-compile the regular expression for the page number
    page_regex = re.compile(r'f(\d+)')

    # Function that parse document queries


    def extract_ark_info(query):

        # Use regular expression to extract the action names directly
        result = re.findall(r'\.([^\.=\?%]*)(?=[\.\?=]|$)', query)

        return set(result) if len(result) > 0 else set('-')


    def extract_page(query):
        # Use the pre-compiled regular expression to search for the page number if not NA else return NA
        page_match = page_regex.search(query)
        return page_match.group(1) if page_match is not None else pd.NA


    ddf['doc_param'] = ddf['endpoint'].where(ddf['is_document'] == 1).map(
        extract_ark_info, meta=('doc_param', 'str'), na_action='ignore')
    ddf['doc_page'] = ddf['endpoint'].where(ddf['is_document'] == 1).map(
        extract_page, meta=('doc_page', 'str'), na_action='ignore')
    ddf['mode'] = ddf['endpoint_4'].where(ddf['is_mode'] == 1)


    # ### Pagination parsing


    # apply extract_page function to the endpoint column
    ddf['pag_param'] = ddf['endpoint'].where(ddf['is_pagination'] == 1).map(
        extract_page, meta=('pag_param', 'str'), na_action='ignore')


    # ## Filtering the dataframe


    # ### Filter unnecessary rows 


    document_exceptions = {'r', 'rk', 'item', 'zoom',
                        'planchecontact', 'vertical', 'double', '-', '--'}

    # Define a function to check if a list of strings has any match with the document exceptions


    def has_document_exception(string):
        return any(exception in string for exception in document_exceptions)


    # fill doc param na values with {-} if is_document is 0 and {--} if is_document is 1
    ddf['doc_param'] = ddf['doc_param'].fillna(
        ddf['is_document'].map({0: '{--}', 1: '{---}'}))

    # drop rows where none of the document actions are present in the set of doc_param column
    ddf = ddf.loc[ddf['doc_param'].apply(lambda x: has_document_exception(x))]

    # replace No_param in doc_param with None
    ddf['doc_param'] = ddf['doc_param'].replace(['{-}', '{--}'], pd.NA)


    # ### Filter unnecessary columns


    # Use the idxmax() method to find the first column with a value of 1 for each row
    ddf['action'] = ddf[actions].idxmax(axis=1)

    # join pag_param and doc_page columns
    ddf['page_number'] = ddf['pag_param'].fillna(ddf['doc_page'])

    # create a list of columns we want to keep
    columns = ['user', 'user_agent', 'country', 'city', 'timestamp',
            'Ark', 'action', 'doc_param', 'page_number', 'mode']

    # Drop the columns that are not needed anymore
    ddf = ddf[columns]


    # ### Bot detection


    # Define a cached version of the parse_ua() function
    @lru_cache(maxsize=None)
    def cached_parse_ua(user_agent):
        return parse_ua(user_agent)


    # Apply the cached_parse_ua() function to the 'user_agent' column
    ddf['is_bot'] = ddf['user_agent'].apply(lambda x: cached_parse_ua(x) == 'Bot')

    # keep the bots in a separate dataframe and merge it with ddf_gallica
    ddf['is_bot'] = ddf['is_bot'] == True | ddf['user_agent'].str.contains(
        'Gallica')

    ddf = ddf.loc[ddf['is_bot'] == False]


    # ## III. Save to parquet

    print(f"computing file {file_number}")
    with ProgressBar():
        result = ddf.compute()
    print(f"file {file_number} computing done")

    print(f"saving file {file_number}")
    with ProgressBar():
        # save to parquet
        result.to_parquet("temp_data/processed_parquet/" + str(file_number) +
                        ".parquet", engine="pyarrow", compression="snappy")
    print(f"file {file_number} saving done")

# %%
# get the file_numbers of the files in the processed_parquet folder
# if the folder is empty, start from 0
if len(os.listdir("temp_data/processed_parquet")) == 0:
    last_file_number = 1
else:
    processed_files_numbers = [int(file.split(".")[0])
                    for file in os.listdir("temp_data/processed_parquet")]
    # get the number of the last file processed
    last_file_number = max(processed_files_numbers)

# process the files starting from the last file processed
if last_file_number == 1:
    restart_file_number = 1
else:
    restart_file_number = last_file_number + 1

for  file_number in range(restart_file_number , len(files_start_indices) + 1):
    # skip computing file 1300 and 1301
    if file_number not in [1300, 1301]:
        
        print(f"processing file {file_number}")
        # process chunck
        print(f"start at index {files_start_indices[file_number - 1]}")
        print(f"stop at index {files_stop_indices[file_number - 1]}")
        process_chunk(files_start_indices[file_number - 1], files_stop_indices[file_number - 1], file_number)
    else:
        print(f"file {file_number} skipped")

print("Done")
