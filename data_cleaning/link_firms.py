from pathlib import Path
import pandas as pd
import re
from unidecode import unidecode

import nama
from nama.config import data_dir as nama_dir
from nama.embedding_similarity import load_similarity_model

from canlobby.config import data_dir

# Use pre-trained string similarity model to predict likely matches
sim = load_similarity_model(Path(nama_dir)/'models'/'nama_base.bin')
sim.to('cuda:0')


# load DPOH names
reg_df = pd.read_csv(Path(data_dir)/'raw_data'/'registrations_enregistrements_ocl_cal'/'Registration_PrimaryExport.csv')

# # Drop rows with missing name (seems like there are some blank rows for some reason)
# reg_df = reg_df.dropna(subset=['EN_FIRM_NM_FIRME_AN','FR_FIRM_NM_FIRME'],how='all')
#

# Take the french version of the name if no english version is available
firms_df = reg_df[['REG_ID_ENR']].copy()
firms_df['firm_raw'] = reg_df['EN_FIRM_NM_FIRME_AN'] \
                        .fillna(reg_df['FR_FIRM_NM_FIRME'])

# Drop any rows that still have a missing name
firms_df = firms_df \
            .drop_duplicates() \
            .dropna()

# load the manually constructed match data
manual = nama.read_csv(Path(data_dir)/'raw_data'/'linking'/'canlobby_train.csv')

matcher = manual.copy()
for c in ['EN_FIRM_NM_FIRME_AN','FR_FIRM_NM_FIRME']:
    matcher = matcher.add_strings(reg_df[c].dropna())


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
fr_en_names_df = reg_df[['EN_FIRM_NM_FIRME_AN','FR_FIRM_NM_FIRME']] \
                    .drop_duplicates() \
                    .dropna() \
                    .query('EN_FIRM_NM_FIRME_AN != FR_FIRM_NM_FIRME')

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
linking_df = firms_df[['REG_ID_ENR','firm_raw']].copy()
linking_df['firm_clean'] = [matcher[s] for s in linking_df['firm_raw']]

linking_df = linking_df[~linking_df['firm_clean'].isin(['Self','Self-employed','None'])]

linking_df.to_csv(Path(data_dir)/'cleaned_data'/'linking'/'firm_registration_linking.csv',index=False)

# Review cases where the raw name differs from the clean name
linking_df.query('firm_raw != firm_clean').sample(50)

matcher.to_df().tail(50)
