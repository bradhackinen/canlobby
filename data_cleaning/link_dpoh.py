from pathlib import Path
import pandas as pd
import re
from unidecode import unidecode
import numpy as np

import nama
from nama.config import data_dir as nama_dir
from nama.embedding_similarity import load_similarity_model

from canlobby import config


# load DPOH names
dpoh_df = pd.read_csv(Path(config.data_dir)/'raw_data'/'communications_ocl_cal'/'Communication_DpohExport.csv')

# Drop rows with missing name (seems like there are some blank rows for some reason)
dpoh_df = dpoh_df.dropna(subset=['DPOH_FIRST_NM_PRENOM_TCPD','DPOH_LAST_NM_TCPD'])

# Create a combined first-last name variable
dpoh_df['dpoh_raw'] = dpoh_df['DPOH_FIRST_NM_PRENOM_TCPD'] + ' ' + dpoh_df['DPOH_LAST_NM_TCPD']

# Keep only the raw name and the comlog_id
dpoh_df = dpoh_df[['COMLOG_ID','dpoh_raw']]


# Create a new matcher for the names
base_matcher = nama.Matcher(dpoh_df['dpoh_raw'])


# Match names based on simple string cleaning
def clean_name(s):
    # Standardize special characters
    s = unidecode(s)

    # Standardize capitalization
    s = s.lower()

    # Remove common titles such as "Honourable", or "Senator"
    s = re.sub(r"\b(the|l')?( right )?hon(\.|ourable)\b",' ',s)
    s = re.sub(r'\bm(rs|r|s|dr)\.? ',' ',s)
    s = re.sub(r'\b(the|mp|minister|min\.|ambassador|brigadier|conservative|national|rear-admiral|commodore|assistant|deputy|executive)\b',' ',s)

    # Standardize whitespace
    s = re.sub(r'\s+',' ',s.strip())

    return s

# clean_name('The Right Honourable John Numbley')
# clean_name('Deputy Minister John Smith')

matcher = base_matcher.unite(clean_name)


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

# Predict matches
predicted_matcher,united_df = embeddings.predict(
                                threshold=0.5,
                                group_threshold=0.5,
                                always_match=base_matcher,
                                never_match=None,
                                return_united=True,
                                )

united_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'dpoh_match_united_pairs.csv',index=False)

# Create linking tables
predicted_matcher.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'dpoh_matches.csv')


dpoh_df['dpoh_clean'] = [predicted_matcher[s] for s in dpoh_df['dpoh_raw']]

dpoh_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'dpoh_com_linking.csv',index=False)



# Create data for review
_,review_df = embeddings.predict(
                                threshold=0.3,
                                group_threshold=0.3,
                                always_match=base_matcher,
                                never_match=None,
                                return_united=True,
                                )

review_df = (review_df
            .query('always_match==False')
            .assign(priority=lambda x: (np.sqrt(x['n_i']*x['n_j'])*x['score']*(1-x['score'])).round(2))
            .sort_values('priority',ascending=False)
            .query('priority > 1')
            .assign(is_match=np.nan))

review_df.to_csv(Path(config.data_dir)/'cleaned_data'/'linking'/'dpoh_match_review.csv',index=False)
