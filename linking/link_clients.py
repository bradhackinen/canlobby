from pathlib import Path
import pandas as pd
import re
from unidecode import unidecode

import nama
from nama.embedding_similarity import load_similarity_model

from canlobby.config import data_dir

# Use pre-trained string similarity model to predict likely matches
sim = load_similarity_model(nama.root_dir/'models'/'nama_base.bin')
sim.to('cuda:0')

# load client names from primary export
com_df = pd.read_csv(Path(data_dir)/'raw_data'/'communications_ocl_cal'/'Communication_PrimaryExport.csv')

# Take the french version of the name if no english version is available
clients_df = com_df[['COMLOG_ID']].copy()
clients_df['client_raw'] = com_df['EN_CLIENT_ORG_CORP_NM_AN'] \
                        .fillna(com_df['FR_CLIENT_ORG_CORP_NM'])

# Drop any rows that still have a missing name
clients_df = clients_df[['COMLOG_ID','client_raw']] \
                    .drop_duplicates() \
                    .dropna()

# load the manually constructed match data
manual = nama.read_csv(Path(data_dir)/'raw_data'/'linking'/'canlobby_train.csv')

matcher = manual.copy()
for c in ['EN_CLIENT_ORG_CORP_NM_AN','FR_CLIENT_ORG_CORP_NM']:
    matcher = matcher.add_strings(com_df[c].dropna())


# Match names based on simple string cleaning
def clean_name(s):
    # Standardize whitespace
    s = re.sub(r'\s+',' ',s.strip())

    # Standardize special characters
    s = unidecode(s)

    # Standardize capitalization
    s = s.title()

    return s


matcher = matcher.unite(clean_name)


# Match french and english versions of each name
fr_en_names_df = com_df[['EN_CLIENT_ORG_CORP_NM_AN','FR_CLIENT_ORG_CORP_NM']] \
                    .drop_duplicates() \
                    .dropna() \
                    .query('EN_CLIENT_ORG_CORP_NM_AN != FR_CLIENT_ORG_CORP_NM')

matcher = matcher.unite(fr_en_names_df.values)


embeddings = sim.embed(matcher)

# Link new strings to similar strings in manual matcher using voronoi clustering
matcher = embeddings.voronoi(
                        seed_strings=manual.strings(),
                        threshold=0.75,
                        base_matcher=matcher)

# Cluster remaining strings while maintaining separation between manual groups
matcher = embeddings.predict(
                        threshold=0.75,
                        group_threshold=0.75,
                        separate_strings=manual.strings(),
                        base_matcher=matcher)


# Create simplified linking table
linking_df = clients_df[['COMLOG_ID','client_raw']].copy()
linking_df['client_clean'] = [matcher[s] for s in linking_df['client_raw']]

linking_df.to_csv(Path(data_dir)/'linking'/'client_comlog_linking.csv')

# Review cases where the raw name differs from the clean name
linking_df.query('client_raw != client_clean').sample(50)
