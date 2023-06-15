import glob
import pandas as pd
from tqdm import tqdm

SESSIONS_PARQUET = 'temp_data/sessions_parquet/*.parquet'

sessions_file_list = glob.glob(SESSIONS_PARQUET)

sessions_df = pd.concat([pd.read_parquet(file) for file in tqdm(sessions_file_list)])

sessions_df.to_parquet('temp_data/sessions_full.parquet')

print('Done!')


