from pathlib import Path
import pandas as pd
import numpy as np

from canlobby.config import data_dir

"""
For some reason, there doesn't seem to be a single ID that links communication
reports to lobbying registrations.

Here we link the two record types by registrant, client number, and date:

- The records must share registrant and client numbers
- Each communication record will be linked to the most recent preceding registration

"""

# load client names from primary export
com_df = pd.read_csv(Path(data_dir)/'raw_data'/'communications_ocl_cal'/'Communication_PrimaryExport.csv')

com_df['COMM_DATE'] = pd.to_datetime(com_df['COMM_DATE'])

# Load registrations from primary export
reg_df = pd.read_csv(Path(data_dir)/'raw_data'/'registrations_enregistrements_ocl_cal'/'Registration_PrimaryExport.csv') \
            .rename(columns={'RGSTRNT_NUM_DECLARANT':'REGISTRANT_NUM_DECLARANT'})

for c in ['EFFECTIVE_DATE_VIGUEUR','END_DATE_FIN']:
    reg_df[c] = pd.to_datetime(reg_df[c])


df = pd.merge(
            com_df[['COMLOG_ID','REGISTRANT_NUM_DECLARANT','CLIENT_ORG_CORP_NUM','COMM_DATE']],
            reg_df[['REG_ID_ENR','REGISTRANT_NUM_DECLARANT','CLIENT_ORG_CORP_NUM','EFFECTIVE_DATE_VIGUEUR','END_DATE_FIN']],
            'left',on=['REGISTRANT_NUM_DECLARANT','CLIENT_ORG_CORP_NUM'])

df['dt'] = np.abs((df['COMM_DATE'] - df['EFFECTIVE_DATE_VIGUEUR']).dt.days)
df['effective'] = (df['EFFECTIVE_DATE_VIGUEUR'] <= df['COMM_DATE'])
df['in_window'] = (df['EFFECTIVE_DATE_VIGUEUR'] <= df['COMM_DATE']) & (df['COMM_DATE'] <= df['END_DATE_FIN'])


df = df.sort_values(['in_window','effective','dt'],ascending=[False,False,True]) \
        .groupby('COMLOG_ID') \
        .first() \
        .reset_index()

df[['in_window','effective','dt']].mean()

df = df[['COMLOG_ID','REG_ID_ENR','dt','effective','in_window']]

df.to_csv(Path(data_dir)/'cleaned_data'/'linking'/'com_reg_linking.csv',index=False)
