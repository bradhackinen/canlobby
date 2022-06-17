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


# load DPOH names
dpoh_df = pd.read_csv(Path(data_dir)/'raw_data'/'communications_ocl_cal'/'Communication_DpohExport.csv')

# Drop rows with missing name (seems like there are some blank rows for some reason)
dpoh_df = dpoh_df.dropna(subset=['DPOH_FIRST_NM_PRENOM_TCPD','DPOH_LAST_NM_TCPD'])

# Create a combined first-last name variable
dpoh_df['dpoh_raw'] = dpoh_df['DPOH_FIRST_NM_PRENOM_TCPD'] + ' ' + dpoh_df['DPOH_LAST_NM_TCPD']

# Create a new matcher for the names
matcher = nama.Matcher(dpoh_df['dpoh_raw'])


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


embeddings = sim.embed(matcher)

pred = embeddings.predict(threshold=0.8)

# Combine predicted matches with existing matches
matcher = matcher.unite(pred)


# Create simplified linking table
linking_df = dpoh_df[['COMLOG_ID','dpoh_raw']].copy()
linking_df['dpoh_clean'] = [matcher[s] for s in linking_df['dpoh_raw']]

linking_df.to_csv(Path(data_dir)/'processed'/'linking'/'dpoh_comlog_linking.csv')

# Review cases where the raw name differs from the clean name
linking_df.query('dpoh_raw != dpoh_clean').sample(50)
