import pandas as pd
import numpy as np

import numpy as np
from datetime import datetime
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#pip install sentence_transformers
#pip install -U scikit-learn
#pip install transformers
######## LOADING EACH DATASET ########
chicago_crimes = pd.read_csv('Chicago_Crimes.csv')
nypd_arrest = pd.read_csv('NYPD_Arrests.csv')
nypd_shooting = pd.read_csv('NYPD_Shooting.csv')
nypd_criminal = pd.read_csv('NYPD_Criminal.csv')
iucr_laws = pd.read_csv('CPD_IUCR.csv')

datasets_preview = {
    'NYS Shooting': nypd_shooting.head(),
    'NYPD Criminal': nypd_criminal.head(),
    'NYPD Arrests': nypd_arrest.head(),
    'Chicago Crimes': chicago_crimes.head(),
    'IUCR Laws': iucr_laws.head()
}

columns_preview = {
    'NYS Shooting': nypd_shooting.columns,
    'NYPD Criminal': nypd_criminal.columns,
    'NYPD Arrests': nypd_arrest.columns,
    'Chicago Crimes': chicago_crimes.columns,
    'IUCR Laws': iucr_laws.columns
}


#dividing the data into two parts of chicago_crimes on the basis of arrest
chicago_arrest=chicago_crimes[chicago_crimes['Arrest']==True].copy()
chicago_criminal=chicago_crimes[chicago_crimes['Arrest']==False].copy()


######## DATA CLEANNING LIKE DATATYPE CHANGING ########
#Chicago Crimes
chicago_arrest['Date'] = pd.to_datetime(chicago_arrest['Date'],format='mixed')
chicago_criminal['Date'] = pd.to_datetime(chicago_criminal['Date'],format='mixed')
chicago_criminal['FBI Code']=chicago_criminal['FBI Code'].astype('str') 
chicago_arrest['FBI Code']=chicago_arrest['FBI Code'].astype('str')
chicago_criminal['IUCR']=chicago_criminal['IUCR'].astype('str') 
chicago_arrest['IUCR']=chicago_arrest['IUCR'].astype('str')


#Nypd shooting 
nypd_shooting['OFNS_Desc']= np.where(nypd_shooting['STATISTICAL_MURDER_FLAG'], 'Shooting and Murdered', 'Shooting and Injured')

#Nypd_Arrests
nypd_arrest['ARREST_DATE']=pd.to_datetime(nypd_arrest['ARREST_DATE'],format='mixed')

#Nypd_Criminal
nypd_criminal['SUMMONS_DATE']=pd.to_datetime(nypd_criminal['SUMMONS_DATE'],format='mixed')

#iucr_laws
iucr_laws['IUCR']=iucr_laws['IUCR'].astype('str')

######### DATA INTERGRATION #########
def get_bert_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Mean pooling and convert to numpy array

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocessing NYPD Arrests descriptions
nypd_arrests_descs = nypd_arrest[['PD_DESC', 'OFNS_DESC']].dropna().drop_duplicates()
nypd_arrests_combined = nypd_arrests_descs.agg(' '.join, axis=1)

# Preprocessing NYPD Criminal descriptions
nypd_criminal_descs = nypd_criminal['OFFENSE_DESCRIPTION'].dropna().drop_duplicates()
nypd_criminal_combined = nypd_criminal_descs

# Preprocessing NYPD Shooting descriptions
nypd_shooting_descs = nypd_shooting['OFNS_Desc'].dropna().drop_duplicates()
nypd_shooting_combined = nypd_shooting_descs

# Preprocessing IUCR Laws descriptions with IUCR Code
iucr_laws_desc = iucr_laws[['IUCR', 'PRIMARY DESCRIPTION', 'SECONDARY DESCRIPTION']].dropna().drop_duplicates()
iucr_laws_combined = iucr_laws_desc[['PRIMARY DESCRIPTION', 'SECONDARY DESCRIPTION']].agg(' '.join, axis=1)

# Generating BERT embeddings
nypd_arrests_embeddings = get_bert_embeddings(nypd_arrests_combined.to_list(), tokenizer, model)
nypd_shooting_embeddings = get_bert_embeddings(nypd_shooting_combined.to_list(), tokenizer, model)
nypd_criminal_embeddings = get_bert_embeddings(nypd_criminal_combined.to_list(), tokenizer, model)
iucr_laws_embeddings = get_bert_embeddings(iucr_laws_combined.to_list(), tokenizer, model)

# Compute cosine similarities
cosine_similarities14 = cosine_similarity(nypd_arrests_embeddings, iucr_laws_embeddings)
cosine_similarities24 = cosine_similarity(nypd_shooting_embeddings, iucr_laws_embeddings)
cosine_similarities34 = cosine_similarity(nypd_criminal_embeddings, iucr_laws_embeddings)

# For each NYPD Law, find the most similar IUCR Law
most_similar_laws14 = cosine_similarities14.argmax(axis=1)
most_similar_laws24 = cosine_similarities24.argmax(axis=1)
most_similar_laws34 = cosine_similarities34.argmax(axis=1)

# Create a DataFrame to display the mapping
mapping_df14 = pd.DataFrame({
    'NYPD Law Description': nypd_arrests_combined.reset_index(drop=True),
    'Most Similar IUCR Law Index': most_similar_laws14,
    'Similarity Score': [cosine_similarities14[i, most_similar_laws14[i]] for i in range(len(most_similar_laws14))]
})
mapping_df24 = pd.DataFrame({
    'NYPD Law Description': nypd_shooting_combined.reset_index(drop=True),
    'Most Similar IUCR Law Index': most_similar_laws24,
    'Similarity Score': [cosine_similarities24[i, most_similar_laws24[i]] for i in range(len(most_similar_laws24))]
})
mapping_df34 = pd.DataFrame({
    'NYPD Law Description': nypd_criminal_combined.reset_index(drop=True),
    'Most Similar IUCR Law Index': most_similar_laws34,
    'Similarity Score': [cosine_similarities34[i, most_similar_laws34[i]] for i in range(len(most_similar_laws34))]
})

# Add descriptions and IUCR codes of the most similar IUCR laws
mapping_df14['IUCR Law Primary Type'] = mapping_df14['Most Similar IUCR Law Index'].apply(
    lambda x: iucr_laws_desc.iloc[x, 1]  if x < len(iucr_laws_desc) else None
)
mapping_df14['IUCR Law Description'] = mapping_df14['Most Similar IUCR Law Index'].apply(
    lambda x:  iucr_laws_desc.iloc[x, 2] if x < len(iucr_laws_desc) else None
)
mapping_df14['IUCR Code'] = mapping_df14['Most Similar IUCR Law Index'].apply(
    lambda x: iucr_laws_desc.iloc[x, 0] if x < len(iucr_laws_desc) else None
)


##shooting IUCR mapping 

mapping_df24['IUCR Law Primary Type'] = mapping_df24['Most Similar IUCR Law Index'].apply(
    lambda x: iucr_laws_desc.iloc[x, 1]  if x < len(iucr_laws_desc) else None
)
mapping_df24['IUCR Law Description'] = mapping_df24['Most Similar IUCR Law Index'].apply(
    lambda x:  iucr_laws_desc.iloc[x, 2] if x < len(iucr_laws_desc) else None
)
mapping_df24['IUCR Code'] = mapping_df24['Most Similar IUCR Law Index'].apply(
    lambda x: iucr_laws_desc.iloc[x, 0] if x < len(iucr_laws_desc) else None
)
#### criminal IUCR mapping
mapping_df34['IUCR Law Primary Type'] = mapping_df34['Most Similar IUCR Law Index'].apply(
    lambda x: iucr_laws_desc.iloc[x, 1]  if x < len(iucr_laws_desc) else None
)
mapping_df34['IUCR Law Description'] = mapping_df34['Most Similar IUCR Law Index'].apply(
    lambda x:  iucr_laws_desc.iloc[x, 2] if x < len(iucr_laws_desc) else None
)
mapping_df34['IUCR Code'] = mapping_df34['Most Similar IUCR Law Index'].apply(
    lambda x: iucr_laws_desc.iloc[x, 0] if x < len(iucr_laws_desc) else None
)


####### IUCR to FBI code Mapping #######

iucr_to_fbi_mapping = chicago_criminal[['IUCR', 'FBI Code']].drop_duplicates()
mapping_df14['IUCR Code']=mapping_df14['IUCR Code'].astype('str')
mapping_df24['IUCR Code']=mapping_df24['IUCR Code'].astype('str')
mapping_df34['IUCR Code']=mapping_df34['IUCR Code'].astype('str')
merged_nypd_arrests_with_fbi = pd.merge(mapping_df14, iucr_to_fbi_mapping, left_on='IUCR Code', right_on='IUCR', how='left').drop('IUCR', axis=1)
merged_nypd_shooting_with_fbi = pd.merge(mapping_df24, iucr_to_fbi_mapping, left_on='IUCR Code', right_on='IUCR', how='left').drop('IUCR', axis=1)
merged_nypd_criminal_with_fbi = pd.merge(mapping_df34, iucr_to_fbi_mapping, left_on='IUCR Code', right_on='IUCR', how='left').drop('IUCR', axis=1)


nypd_shooting_full = pd.merge(nypd_shooting, 
                              merged_nypd_shooting_with_fbi[['NYPD Law Description', 'IUCR Code', 'FBI Code','IUCR Law Primary Type','IUCR Law Description']], 
                              left_on='OFNS_Desc', 
                              right_on='NYPD Law Description',
                              how='left')
nypd_criminal_full = pd.merge(nypd_criminal, 
                              merged_nypd_criminal_with_fbi[['NYPD Law Description', 'IUCR Code', 'FBI Code','IUCR Law Primary Type','IUCR Law Description']], 
                              left_on='OFFENSE_DESCRIPTION', 
                              right_on='NYPD Law Description',
                              how='left')
nypd_arrest['Combined Description'] = nypd_arrest['PD_DESC']+' '+nypd_arrest['OFNS_DESC']
nypd_arrests_full = pd.merge(nypd_arrest, 
                             merged_nypd_arrests_with_fbi[['NYPD Law Description', 'IUCR Code', 'FBI Code','IUCR Law Primary Type','IUCR Law Description']], 
                             left_on=['Combined Description'], 
                             right_on='NYPD Law Description',
                             how='left')
chicago_criminal_full = pd.merge(chicago_criminal, 
                             iucr_laws[['IUCR','PRIMARY DESCRIPTION','SECONDARY DESCRIPTION']], 
                             left_on=['IUCR','Primary Type'], 
                             right_on=['IUCR','PRIMARY DESCRIPTION'],
                             how='left')
chicago_arrest_full = pd.merge(chicago_arrest, 
                             iucr_laws[['IUCR','PRIMARY DESCRIPTION','SECONDARY DESCRIPTION']], 
                             left_on=['IUCR','Primary Type'], 
                             right_on=['IUCR','PRIMARY DESCRIPTION'],
                             how='left')

chicago_arrest_full['City']='Chicago'
chicago_criminal_full['City']='Chicago'
nypd_arrests_full['City']='New York'
nypd_criminal_full['City']='New York'
nypd_shooting_full['City']='New York'



chicago_arrest_full=chicago_arrest_full[["Date", "Description", "PRIMARY DESCRIPTION", "SECONDARY DESCRIPTION", "FBI Code", "IUCR", "Latitude", "Longitude", "City"]].rename(columns={
    'Date': 'Arrest_Date',
    'Description': 'OFFENSE_DESCRIPTION',
    'PRIMARY DESCRIPTION': 'Primary_Offense_Description',
    'SECONDARY DESCRIPTION': 'Second_Offense_Description',
    "IUCR":"IUCR_Code"
}).drop_duplicates()

nypd_arrests_full=nypd_arrests_full[["ARREST_DATE", "OFNS_DESC", "IUCR Law Primary Type", "IUCR Law Description", "FBI Code", "IUCR Code", "Latitude", "Longitude", "City"]].rename(columns={
    'ARREST_DATE': 'Arrest_Date',
    'OFNS_DESC': 'OFFENSE_DESCRIPTION',
    'IUCR Law Primary Type': 'Primary_Offense_Description',
    'IUCR Law Description': 'Second_Offense_Description',
    'IUCR Code':'IUCR_Code'
}).drop_duplicates()

nypd_shooting_full=nypd_shooting_full[["OCCUR_DATE", "OFNS_Desc", "IUCR Law Primary Type", "IUCR Law Description", "FBI Code", "IUCR Code", "Latitude", "Longitude", "City"]].rename(columns={
    'OCCUR_DATE': 'Arrest_Date',
    'OFNS_Desc': 'OFFENSE_DESCRIPTION',
    'IUCR Law Primary Type': 'Primary_Offense_Description',
    'IUCR Law Description': 'Second_Offense_Description',
    'IUCR Code':'IUCR_Code'
    
}).drop_duplicates()

# Concatenating while selecting only the specified columns for master_crime

chicago_criminal_full=chicago_criminal_full[["Date", "Description", "PRIMARY DESCRIPTION", "SECONDARY DESCRIPTION", "FBI Code", "IUCR", "Latitude", "Longitude", "City"]].rename(columns={
    'Date': 'Crime_Date',
    'Description': 'OFFENSE_DESCRIPTION',
    'PRIMARY DESCRIPTION': 'Primary_Offense_Description',
    'SECONDARY DESCRIPTION': 'Second_Offense_Description',
    "IUCR":"IUCR_Code"
}).drop_duplicates()

nypd_criminal_full=nypd_criminal_full[["SUMMONS_DATE", "OFFENSE_DESCRIPTION", "IUCR Law Primary Type", "IUCR Law Description", "FBI Code", "IUCR Code", "Latitude", "Longitude", "City"]].rename(columns={
    'SUMMONS_DATE': 'Crime_Date',
    'OFFENSE_DESCRIPTION': 'OFFENSE_DESCRIPTION',
    'IUCR Law Primary Type': 'Primary_Offense_Description',
    'IUCR Law Description': 'Second_Offense_Description',
    'IUCR Code':'IUCR_Code' 
}).drop_duplicates()



master_arrest=pd.concat([chicago_arrest_full,nypd_arrests_full,nypd_shooting_full])
master_crime=pd.concat([chicago_criminal_full,nypd_criminal_full])

# Printing null counts and the dataframes
print(master_arrest.isnull().sum())
print(master_crime.isnull().sum())
## writing down the result into two Dataset
master_arrest['Arrest_Date']=master_arrest['Arrest_Date'].astype('str')
master_crime['Crime_Date']=master_crime['Crime_Date'].astype('str')
master_arrest.to_csv('Master_Arrest')
master_crime.to_csv('Master_Crime')