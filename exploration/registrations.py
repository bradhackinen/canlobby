from pathlib import Path
import pandas as pd

from canlobby.config import data_dir

# load DPOH names
reg_df = pd.read_csv(Path(data_dir)/'raw_data'/'registrations_enregistrements_ocl_cal'/'Registration_PrimaryExport.csv')

reg_df['year'] = reg_df['EFFECTIVE_DATE_VIGUEUR'].str.slice(0,4)

reg_df['year'].value_counts() \
    .sort_index() \
    .plot()


reg_df.groupby('year')['RGSTRNT_NUM_DECLARANT'].nunique() \
    .plot()

reg_df.groupby('year')['CLIENT_ORG_CORP_NUM'].nunique() \
    .plot()

reg_df.groupby('year')['EN_FIRM_NM_FIRME_AN'].nunique() \
    .plot()


reg_df.head(1).T
