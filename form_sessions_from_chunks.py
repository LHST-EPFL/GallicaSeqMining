# %%
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import swifter
import re
import os
from collections import defaultdict
# deactivating warnings
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## I. Read preprocessed data

# %%
NUMBER_OF_FILES = 30
INACTIVE_THRESHOLD = 60
REQUEST_THRESHOLD = 1
IS_BOT = 0

# if "temp_data/sessions_parquet" does not exist, create it
if not os.path.exists("temp_data/sessions_parquet"):
    os.makedirs("temp_data/sessions_parquet")

# if the folder is empty, set the last process file number to 1
if len(os.listdir("temp_data/processed_parquet")) == 0:
    last_process_file_number = 1
else:
    processed_files_numbers = [int(file.split(".")[0])
                    for file in os.listdir("temp_data/processed_parquet")]
    # get the number of the last file processed
    last_process_file_number = max(processed_files_numbers)

# if the folder is empty, set the last session file number to 1
if len(os.listdir("temp_data/sessions_parquet")) == 0:
    last_session_file_number = 1
else:
    session_files_numbers = [int(file.split("_")[1].split(".")[0])
                    for file in os.listdir("temp_data/sessions_parquet")]
    # get the number of the last file processed
    last_session_file_number = max(session_files_numbers)

def form_sessions(file_number):
    # read data
    df = pd.read_parquet("temp_data/processed_parquet/" + str(file_number) + ".parquet", engine="pyarrow")

 
    if IS_BOT:
        df = df[df["is_bot"] == 1]
    else:
        df = df[df["is_bot"] == 0]

    df = df.drop(columns=["is_bot"]).reset_index(drop=True)


    # ## II. Detect sessions


    df['doc_param'] = df['doc_param'].fillna('No_param')


    # Convert the timestamp column to a datetime data type, and set it as the index
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%d/%b/%Y:%H:%M:%S %z")

    # Sort the DataFrame by user ID and timestamp
    df.sort_values(["user", "timestamp"], inplace=True)

    # filter users with less than 5 requests in total
    df = df.groupby("user").filter(lambda x: len(x) >= 5)

    # Calculate the time difference between consecutive log entries for each user ID
    df["time_diff"] = df.groupby("user")["timestamp"].diff()

    # Set a threshold for determining the start of a new session (e.g., 30 minutes)
    threshold = pd.Timedelta(minutes=INACTIVE_THRESHOLD)

    # Identify the first ever request by each user
    df["first_request"] = df.groupby("user").cumcount() == 0

    # Determine the start of each session
    df["session_start"] = df["first_request"] | (df["time_diff"] > threshold)

    # Drop the first_request column
    df.drop("first_request", axis=1, inplace=True)

    # Calculate the session number for each user
    df["session_number"] = df.groupby("user")["session_start"].cumsum()

    # calculate the position of each request in the session
    df["request_position_in_session"] = df.groupby(
        ["user", "session_number"]).cumcount() + 1


    # ## III. Remove sessions with very frequent requests (Non human behavior)



    # Combine groupby operations and use agg() function to calculate aggregated values in a single pass
    aggregated_data = df.groupby(["user", "session_number"]).agg(
        request_count=("request_position_in_session", "count"),
        total_time=(
            "time_diff", lambda x: x[x.index[0] != x.index].sum().total_seconds())
    )

    # Calculate request frequencies
    aggregated_data["request_frequency_in_session"] = aggregated_data["request_count"] / \
        aggregated_data["total_time"]
    aggregated_data["request_frequency_global"] = aggregated_data.groupby(
        "user")["request_frequency_in_session"].transform("mean")

    # Mark users with request_frequency_global > REQUEST_THRESHOLD
    aggregated_data["is_marked"] = aggregated_data["request_frequency_global"] > REQUEST_THRESHOLD

    # Merge the is_marked column back to the original DataFrame
    df = df.merge(aggregated_data["is_marked"], left_on=[
                "user", "session_number"], right_index=True)

    # Filter marked and unmarked users
    df = df.query("is_marked == False").drop("is_marked", axis=1)


    # ## IV. Recompute sessions


    # recompute the session number for each user
    df["session_number"] = df.groupby("user")["session_start"].cumsum()

    # recompute the position of each request in the session
    df["request_position_in_session"] = df.groupby(
        ["user", "session_number"]).cumcount() + 1

    # Generate a user-friendly session ID
    df["session_id"] = (
        "S_"
        + str(file_number)
        + "_"
        + df["session_number"].astype(str)
        + "_U_"
        + df["user"].astype(str)
    )

    # Generate a user-friendly request ID
    df["request_id"] = (
        "S_"
        + df["session_number"].astype(str)
        + "_"
        + df["request_position_in_session"].astype(str)
        + "_U_"
        + df["user"].astype(str)
    )

    # sort by user than by session number than by session position
    df.sort_values(["user", "session_number",
                "request_position_in_session"], inplace=True)

    # mark the end of each session (should be true for the last request of each session and false for the rest)
    df["session_end"] = df.groupby(["user", "session_number"])[
        "request_position_in_session"].transform("max") == df["request_position_in_session"]

    # Convert time difference to seconds
    df["time_diff"] = df["time_diff"].apply(lambda x: x.total_seconds())


    # ## V. Action tree


    # ### Index documents and document pages in sessions


    df['page_number'] = df['page_number'].fillna(-999).astype(int)

    # filter rows where action is pagination and no page number is provided
    df = df[~((df["action"] == "is_pagination") & (df["page_number"] == -999))]

    # compute the document number in the session
    df['doc_number_in_session'] = df.groupby('session_id')['Ark'].transform(
        lambda x: x.dropna().map(dict(zip(x.dropna().unique(), range(1, len(x.dropna().unique())+1)))))
    df['doc_number_in_session'] = df['doc_number_in_session'].fillna(
        -999).astype(int)

    # in each session, in each row compute the previous page in the same document (if exists)
    df['prev_page_in_doc'] = df.groupby(['session_id', 'doc_number_in_session'])[
        'page_number'].shift(1)
    df.loc[(df['action'] != 'is_document') & (df['action']
                                            != 'is_pagination'), 'prev_page_in_doc'] = np.nan
    df['prev_page_in_doc'] = df['prev_page_in_doc'].fillna(-999).astype(int)

    # filter the rows where the page number and the previous page in the same document are the same and not -999
    df = df.loc[(df['page_number'] != df['prev_page_in_doc']) | (
        df['page_number'] == -999) & (df['prev_page_in_doc'] == -999)]


    # is document acess if the action is is_document and the previous page is -999
    df['is_document_access'] = (df['action'] == 'is_document') & (
        df['prev_page_in_doc'] == -999).astype(int)

    # is homepage if the action is is_homepage
    df['is_homepage'] = (df['action'] == 'is_homepage').astype(int)

    # is heading if the action is is_heading
    df['is_heading_navigation'] = (df['action'] == 'is_heading').astype(int)

    # is blog if the action is is_blog
    df['is_blog_navigation'] = (df['action'] == 'is_blog').astype(int)

    # is first page if the previous page is -999 and the page number is different from -999
    df['is_first_page'] = (df['prev_page_in_doc'] == -
                        999) & (df['page_number'] != -999) & (df['is_document_access'] == False).astype(int)
    # is next page if the page number is the previous page + 1 or + 2 (for the case of double page)
    df['is_next_page'] = (df['page_number'] == df['prev_page_in_doc'] +
                        1) | (df['page_number'] == df['prev_page_in_doc'] + 2).astype(int)
    # is prev page if the page number is the previous page - 1 or - 2 (for the case of double page)
    df['is_prev_page'] = (df['page_number'] == df['prev_page_in_doc'] -
                        1) | (df['page_number'] == df['prev_page_in_doc'] - 2).astype(int)
    # is chosen page if the page number is different from -999 and the previous page is different from -999 and the difference betweem them is greater than 2
    df['is_chosen_page'] = (df['page_number'] != -999) & (df['prev_page_in_doc']
                                                        != -999) & (abs(df['page_number'] - df['prev_page_in_doc']) > 2).astype(int)
    df['is_revisit_document'] = (df['action'] == 'is_document') & (
        df['page_number'] == -999) & (df['prev_page_in_doc'] != -999).astype(int)


    # is to single page mode if mode is single
    df['is_to_single_page_mode'] = (
        df['mode'] == 'SINGLE').fillna(False).astype(int)
    # is to double page mode if mode is double
    df['is_to_double_page_mode'] = (
        df['mode'] == 'DOUBLE').fillna(False).astype(int)
    # is to multi page mode if mode is multi
    df['is_to_multi_page_mode'] = (df['mode'] == 'MULTI').fillna(False).astype(int)
    # is to vertical page mode if mode is vertical
    df['is_to_vertical_page_mode'] = (
        df['mode'] == 'VERTICAL').fillna(False).astype(int)
    # is to audio page mode if mode is audio
    df['is_to_audio_page_mode'] = ((df['mode'] == 'AUDIO') | (df['mode'] == 'TEXT_RAW') | (df['mode'] == 'MEDIA') | (df['mode'] == 'D3')).fillna(False).astype(int)
    # is zoom  if the action is is_zoom
    df['is_zoom'] = ((df['action'] == 'is_zoom') | (df['mode'] == 'ZOOM')).fillna(False).astype(int)

    # is simple search if the action is is_simple_search
    df['is_simple_search'] = (df['action'] == 'is_simple_search').astype(int)
    # is advanced search if the action is is_advanced_search
    df['is_advanced_search'] = (df['action'] == 'is_advanced_search').astype(int)
    # is filtering search results if the action is is_filtering_search_results
    df['is_filtering_search_results'] = (
        df['action'] == 'is_filtering_search_results').astype(int)

    # is document download if the action is is_document_download
    df['is_document_download'] = (
        df['action'] == 'is_document_download').astype(int)
    # is page download if the action is is_page_download
    df['is_page_download'] = (df['action'] == 'is_page_download').astype(int)


    precise_actions = ['is_document_access', 'is_homepage', 'is_heading_navigation', 'is_blog_navigation', 'is_first_page', 'is_next_page', 'is_prev_page', 'is_chosen_page', 'is_revisit_document', 'is_to_single_page_mode', 'is_to_double_page_mode',
                    'is_to_multi_page_mode', 'is_to_vertical_page_mode', 'is_to_audio_page_mode', 'is_zoom', 'is_simple_search', 'is_advanced_search', 'is_filtering_search_results', 'is_document_download', 'is_page_download']

    df['is_sum'] = df[precise_actions].sum(axis=1)

    # make sure that no request has more than one action
    if df[df['is_sum'] > 1].shape[0] != 0:
        # print the requests that have more than one action
        print(df[df['is_sum'] > 1].shape)
        print(df.loc[df['is_sum'] > 1, ['request_id', 'action', 'mode']])
        # raise an error
        raise AssertionError("There are requests that have more than one action")
    assert df[df['is_sum'] > 1].shape[0] == 0
    
    # make sure that no request has no action
    if df[df['is_sum'] < 1].shape[0] != 0:
        # print the requests that have no action
        print(df.loc[df['is_sum'] < 1, ['request_id', 'action', 'mode']])
        # raise an error
        raise AssertionError("There are requests that have no action")
    assert df[df['is_sum'] < 1].shape[0] == 0

    # precise request
    df["precise_action"] = df[precise_actions].astype(int).idxmax(axis=1)
    # remove the is in the name of the action
    df["precise_action"] = df["precise_action"].str.replace("is_", "")


    # ### Build sessions as sequences of actions


    # build the sessions as sequences of actions including the timestamp of the action (rename the column to be more clear)
    sessions = df[['session_id', 'precise_action', 'timestamp', 'Ark']]
    sessions = sessions.rename(columns={'precise_action': 'action'})


    # Save as Parquet
    if ~IS_BOT:
        sessions.to_parquet("temp_data/sessions_parquet/sessions_" + str(file_number) +
                            ".parquet", engine="pyarrow", index=False, compression="snappy")
        print("Saved sessions_" + str(file_number) + ".parquet")
    
# form the sessions the files starting from the last sessions file processed
if last_session_file_number == 1:
    restart_session_file_number = 1
else:
    restart_session_file_number = last_session_file_number + 1    

for file_number in range(restart_session_file_number, last_process_file_number + 1):
    form_sessions(file_number)

print("Done")


