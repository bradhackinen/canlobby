from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import re

from canlobby.config import data_dir

# Load raw data files
com_df = pd.read_csv(Path(data_dir)/'raw_data'/'communications_ocl_cal'/'Communication_PrimaryExport.csv')
dpoh_df = pd.read_csv(Path(data_dir)/'raw_data'/'communications_ocl_cal'/'Communication_DpohExport.csv')

# Load linking files
com_reg_linking_df = pd.read_csv(Path(data_dir)/'cleaned_data'/'linking'/'com_reg_linking.csv')
client_linking_df = pd.read_csv(Path(data_dir)/'cleaned_data'/'linking'/'client_com_linking.csv')
dpoh_linking_df = pd.read_csv(Path(data_dir)/'cleaned_data'/'linking'/'dpoh_com_linking.csv')
firm_linking_df = pd.read_csv(Path(data_dir)/'cleaned_data'/'linking'/'firm_reg_linking.csv')

# Build lobbyist-firm-dpoh contact data
contacts_df = com_df.copy()
contacts_df = pd.merge(contacts_df,com_reg_linking_df,'left',on='COMLOG_ID')
contacts_df = pd.merge(contacts_df,client_linking_df,'left',on='COMLOG_ID')
contacts_df = pd.merge(contacts_df,dpoh_linking_df,'left',on='COMLOG_ID')
contacts_df = pd.merge(contacts_df,dpoh_df,'left',on='COMLOG_ID')
contacts_df = pd.merge(contacts_df,firm_linking_df,'left',on='REG_ID_ENR')

contacts_df = contacts_df.rename(columns={
                    'client_clean':'client',
                    'firm_clean':'firm',
                    'dpoh_clean':'dpoh_name',
                    'DPOH_TITLE_TITRE_TCPD':'dpoh_title',
                    'INSTITUTION':'dpoh_institution',
                    'BRANCH_UNIT_DIRECTION_SERVICE':'dpoh_branch',
                    'COMM_DATE':'date',
                    'COMLOG_ID':'comlog_id',
                    })

contacts_df['registrant_name'] = contacts_df['RGSTRNT_1ST_NM_PRENOM_DCLRNT'] + ' ' + contacts_df['RGSTRNT_LAST_NM_DCLRNT']
contacts_df['registrant_id'] = contacts_df['REGISTRANT_NUM_DECLARANT']
contacts_df['registration_type'] = contacts_df['REG_TYPE_ENR'].replace({1:'Consultant',2:'In-house Corporation',3:'In-house Organization'})
contacts_df['dpoh_is_mp'] = [bool(re.match(r'(MP|Member of Parliament|Minister)',s)) for s in contacts_df['dpoh_title'].astype(str)]

# Select cleaned columns
contacts_df = contacts_df[[
                'comlog_id',
                'date',
                'registrant_id',
                'registrant_name',
                'registration_type',
                'client',
                'firm',
                'dpoh_name',
                'dpoh_title',
                'dpoh_institution',
                'dpoh_branch',
                'dpoh_is_mp',
                ]]


contacts_df.to_csv(Path(data_dir)/'cleaned_data'/'contacts.csv',index=False)


def mode(x):
    v = pd.value_counts(x)
    if len(v):
        return v.index.values[0]
    else:
        return np.nan
    
# Compile annual contact summary
annual_contacts_df = contacts_df \
                    .assign(year=contacts_df['date'].str.slice(0,4).astype(int)) \
                    .sort_values('date') \
                    .groupby(['year','registrant_id','registration_type','client','dpoh_name','dpoh_institution']) \
                    .agg({
                        'comlog_id':'nunique',
                        'registrant_name':mode,
                        'firm':mode,
                        'dpoh_title':mode,
                        'dpoh_branch':mode,
                        'dpoh_is_mp':mode}) \
                    .reset_index() \
                    .rename(columns={'comlog_id':'n'})

annual_contacts_df.to_csv(Path(data_dir)/'cleaned_data'/'annual_contacts.csv',index=False)