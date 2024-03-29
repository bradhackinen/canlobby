from pathlib import Path
import pandas as pd
import re
from unidecode import unidecode
import json
import numpy as np

import nama
from nama.config import data_dir as nama_dir
from nama.embedding_similarity import load_similarity_model

from canlobby import config


base_matcher = nama.Matcher()

# Load client and firm names from the registration files
reg_df = pd.read_csv(Path(config.data_dir)/'raw_data'/'registrations_enregistrements_ocl_cal'/'Registration_PrimaryExport.csv')
reg_df.head(1).T
for c in ['CLIENT_ORG_CORP_NM','FIRM_NM_FIRME']:
    base_matcher = (base_matcher
               .add_strings(reg_df[f'EN_{c}_AN'].dropna())
               .add_strings(reg_df[f'FR_{c}'].dropna()))

    # Match french and english versions of each client name
    translations_df = reg_df[[f'EN_{c}_AN',f'FR_{c}']] \
                        .drop_duplicates() \
                        .dropna() \
                        .query(f'EN_{c}_AN != FR_{c}')

    base_matcher = base_matcher.unite(translations_df.values)


# load client names from communication records (just in case they are different)
com_df = pd.read_csv(Path(config.data_dir)/'raw_data'/'communications_ocl_cal'/'Communication_PrimaryExport.csv')

# Add both French and English versions of the name
for c in ['EN_CLIENT_ORG_CORP_NM_AN','FR_CLIENT_ORG_CORP_NM']:
    base_matcher = base_matcher.add_strings(com_df[c].dropna())

# Match french and english versions of each client name
translations_df = com_df[['EN_CLIENT_ORG_CORP_NM_AN','FR_CLIENT_ORG_CORP_NM']] \
                    .drop_duplicates() \
                    .dropna() \
                    .query('EN_CLIENT_ORG_CORP_NM_AN != FR_CLIENT_ORG_CORP_NM')

base_matcher = base_matcher.unite(translations_df.values)


# Match names based on simple string cleaning
def clean_name(s):
    # Standardize whitespace
    s = re.sub(r'\s+',' ',s.strip())

    # Standardize special characters
    s = unidecode(s)

    # Standardize capitalization
    s = s.title()

    return s

base_matcher = base_matcher.unite(clean_name)


manual_pairs_df = pd.read_csv(Path(config.data_dir)/'raw_data'/'linking'/'org_match_review_SR_Mar4.csv')

# Add manual alway_match information
with open(Path(config.data_dir)/'raw_data'/'linking'/'canlobby_always_match.json','r') as f:
    always_match = json.load(f)

# Add manual matches to base matcher
base_matcher = base_matcher.unite(always_match)


always_match_pairs_df = manual_pairs_df[~manual_pairs_df['is_match'].isin(['n','?'])]
never_match_pairs_df = manual_pairs_df[manual_pairs_df['is_match']=='n']

base_matcher = base_matcher.unite(always_match_pairs_df[['i','j']].values)



# Add manual never_match information
with open(Path(config.data_dir)/'raw_data'/'linking'/'canlobby_never_match.json','r') as f:
    never_match = json.load(f)

never_match += [list(pair) for pair in never_match_pairs_df[['i','j']].values]



# Add additional org names
alt_names_df = pd.read_csv(Path(config.data_dir)/'raw_data'/'linking'/'orgs_update_SR_Mar4.csv')

for c in ['name','name_update']:
    base_matcher = base_matcher.add_strings(alt_names_df[c])
base_matcher = base_matcher.unite(alt_names_df[['name','name_update']].values)

# Use pre-trained string similarity model to predict likely matches
sim = load_similarity_model(Path(nama_dir)/'models'/'nama_base.bin')
sim.to(config.nama_device)

# Embed the strings
embeddings = sim.embed(base_matcher,device='cpu')

# Swap the similarity model and embeddings on the GPU
sim.to('cpu')
embeddings.to(config.nama_device)

if config.nama_fp16:
    # Use 16bit floats for higher speed (shouldn't affect accuracy)
    embeddings.half()

predicted_matcher,united_df = embeddings.predict(
                                threshold=0.5,
                                group_threshold=0.5,
                                always_match=base_matcher,
                                never_match=never_match,
                                return_united=True,
                                always_never_conflicts='ignore'
                                )

united_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'org_match_united_pairs.csv',index=False)

# Create linking tables
predicted_matcher.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'org_matches.csv')


# Create client-registration linking table
client_reg_df = reg_df[['REG_ID_ENR']].copy()
client_reg_df['client_raw'] = (reg_df['EN_CLIENT_ORG_CORP_NM_AN']
                                .fillna(reg_df['FR_CLIENT_ORG_CORP_NM']))
client_reg_df = client_reg_df.dropna()
client_reg_df['client_clean'] = [predicted_matcher[s] for s in client_reg_df['client_raw']]

client_reg_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'client_reg_linking.csv',index=False)

# Create firm-registration linking table
firm_reg_df = reg_df[['REG_ID_ENR']].copy()
firm_reg_df['firm_raw'] = (reg_df['EN_FIRM_NM_FIRME_AN']
                                .fillna(reg_df['FR_FIRM_NM_FIRME']))
firm_reg_df = firm_reg_df.dropna()
firm_reg_df['firm_clean'] = [predicted_matcher[s] for s in firm_reg_df['firm_raw']]

firm_reg_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'firm_reg_linking.csv',index=False)

# Create client-communication linking table
client_com_df = com_df[['COMLOG_ID']].copy()
client_com_df['client_raw'] = (com_df['EN_CLIENT_ORG_CORP_NM_AN']
                                .fillna(com_df['FR_CLIENT_ORG_CORP_NM']))
client_com_df = client_com_df.dropna()
client_com_df['client_clean'] = [predicted_matcher[s] for s in client_com_df['client_raw']]

client_com_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'client_com_linking.csv',index=False)



# Create data for review
_,review_df = embeddings.predict(
                                threshold=0.3,
                                group_threshold=0.3,
                                always_match=base_matcher,
                                never_match=never_match,
                                return_united=True,
                                always_never_conflicts='ignore'
                                )

review_df = (review_df
            .query('always_match==False')
            .assign(priority=lambda x: (np.sqrt(x['n_i']*x['n_j'])*x['score']*(1-x['score'])).round(2))
            .query('score < 0.8')
            .assign(is_match=np.nan))

review_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'org_match_review.csv',index=False)
