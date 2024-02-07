#Install basic python pcakges
import pandas as pd
import numpy as np

import pubchempy as pcp
from pubchempy import get_compounds, Compound
from collections import Counter

# Molecular Descriptors Calculation
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors

# Modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from pandas.core.common import random_state
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from keras.models import load_model

def ai_qsar_predict(data):
    # data = [[CAS, Species]]
    df = pd.DataFrame(data, columns=['CAS', 'Species'])

    # df = pd.read_csv('HL6_one.csv')
    # df = df[:1]
    
    df_with_cids = cas_to_cid(df)
    df_with_smiles = cid_to_smiles(df_with_cids)

    rdkit_descrs = cal_rdkit_descr(df_with_smiles['SMILES']).drop(['BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW'],axis=1)

    ECFP6_descrs = cal_ECFP6_descr(df_with_smiles['SMILES'])

    FCFP6_descrs = cal_FCFP6_descr(df_with_smiles['SMILES'])

    MACCS_descrs = cal_MACCS_descr(df_with_smiles['SMILES'])

    # print(df['Species'])

    species_descr = pd.get_dummies(df['Species'], dtype=float)

    # print(species_descr)

    # Define a list of all possible species
    all_species = ['Chickens', 'Goats', 'Sheep', 'Swine', 'Cattle', 'Turkeys']

    # Add columns for the missing species with all 0's
    missing_species = [species for species in all_species if species not in species_descr.columns]
    for species in missing_species:
        species_descr[species] = 0

    species_descr = species_descr[all_species]

    scaler = MinMaxScaler()
    X_rdkit_descrs = rdkit_descrs
    X_rdkit_descrs_scal = scaler.fit_transform(X_rdkit_descrs)
    X_rdkit_descrs_scal = pd.DataFrame(X_rdkit_descrs_scal, columns = rdkit_descrs.columns.values.tolist())

    X_rdkit = pd.concat([X_rdkit_descrs_scal.reset_index(drop=True)], axis = 1)
    X_ECFP6 = pd.concat([ECFP6_descrs.reset_index(drop=True)], axis = 1)
    X_FCFP6 = pd.concat([FCFP6_descrs.reset_index(drop=True)], axis = 1)
    X_MACCS = pd.concat([MACCS_descrs.reset_index(drop=True)], axis = 1)
    All_descr = pd.concat([species_descr, X_rdkit_descrs_scal.reset_index(drop=True), X_ECFP6.reset_index(drop=True), X_FCFP6.reset_index(drop=True), X_MACCS.reset_index(drop=True)], axis = 1)
    # y = df['LambdaZHl']

    best_model_v2 = load_model('best_model_all.h5')

    y_preds_v2 = best_model_v2.predict(All_descr)

    return y_preds_v2

'''
This script enables automatically connecting to the PubChem database,
Transfer of CAS numbers which are converted to CID identifiers
as first step and then resolved to respective SMILES codes.

'''

def cas_to_cid (df):

  df['PubChemCID'] = None
  # Searching PubChem for CID
  for i, CAS in df['CAS'].items():
    try:
        # Searching PubChem for CID using CAS number
        results = get_compounds(CAS, 'name')
        if results:
            cid = results[0].cid
            df.at[i, 'PubChemCID'] = cid
        else:
            print(f"No results found for CAS number {CAS}")

    except Exception as e:
        pass # silent the error message; otherwise it will print a lot error message: server is busy

  # Check PubChemCID; If CID is None, run the "While loop" and request PubChem server again to get all Cid
  while df['PubChemCID'].isnull().sum() > 0:
    print (df['PubChemCID'].isnull().sum())
    for i, CAS in df[df['PubChemCID'].isnull()]['CAS'].items():
      try:
        # Searching PubChem for CID using CAS number
        results = get_compounds(CAS, 'name')
        if results:
            cid = results[0].cid
            df.at[i, 'PubChemCID'] = cid
        else:
            print(f"No results found for CAS number {CAS}")
      except Exception as e:
        pass

  return df

# Function for searching and extracting SMILES code with entering CID
def cid_to_smiles (data):
  data['SMILES'] = None

  for i, cid in data['PubChemCID'].items():
    try:
      compound = pcp.Compound.from_cid(cid)
      if compound:
        smiles = compound.canonical_smiles
        data.at[i, 'SMILES'] = smiles
      else:
        print(f'No results found for PubChemCID {cid}')

    except Exception as e:
        pass # silent the error message; otherwise it will print a lot error message: "PUG-REST server is busy"

  # Check SMILES; If SMILES is None, run the "While loop" and request PubChem server again to get all SMILES
  while data['SMILES'].isnull().sum() > 0:
    print (data['SMILES'].isnull().sum())
    for i, cid in data[data['SMILES'].isnull()]['PubChemCID'].items():
     try:
       compound = pcp.Compound.from_cid(cid)
       if compound:
        smiles = compound.canonical_smiles
        data.at[i, 'SMILES'] = smiles
       else:
        print(f'No results found for PubChemCID {cid}')
     except Exception as e:
       pass

  return data


# Define a function that transform SMILES string into RDKIT descriptors
def cal_rdkit_descr(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)

    return pd.DataFrame(mol_descriptors, index=smiles, columns=desc_names)


# Define a funciton that transform a SMILES string into an FCFP (if use_features = TRUE) or--
# --the Extended-Connectivity Fingerprints (ECFP) descriptors (if use_features = FALSE)

def cal_ECFP6_descr(smiles,
            R = 3,
            nBits = 2**10, # nBits = 1024
            use_features = False,
            use_chirality = False):

   '''
   Inputs:
   - smiles...SMILES string of input compounds
   - R....Maximum radius of circular substructures--By using this radius parameter, we compute ECFP6 (the equivalent of radius 3)
   - nBits....number of bits, default is 2048. 1024 is also widely used.
   - use_features...if true then use pharmacophoric atom features (FCFPs), if false then use stadnard DAYLIGHT atom features (ECFP)
   - use_chirality...if true then append tetrahedral chirality flags to atom features
   Outputs:
   - pd.DataFrame...ECFP or FCFPs with length nBits and maximum radus R

   '''
   mols = [AllChem.MolFromSmiles(i) for i in smiles]

   ecfp_descriptors = []
   for mol in mols:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol,
                                radius = R,
                                nBits = nBits,
                                useFeatures = use_features,
                                useChirality = use_chirality)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(ecfp, array)
        ecfp_descriptors.append(ecfp)

   return pd.DataFrame([list(l) for l in ecfp_descriptors], index = smiles, columns=[f'ECFP6_Bit_{i}' for i in range(nBits)])


# Define a funciton that transform a SMILES string into an FCFP (if use_features = TRUE)
def cal_FCFP6_descr(smiles,
            R = 3,
            nBits = 2**10, # nBits = 1024
            use_features = True,
            use_chirality = False):

   mols = [AllChem.MolFromSmiles(i) for i in smiles]

   fcfp_descriptors = []
   for mol in mols:
        fcfp = AllChem.GetMorganFingerprintAsBitVect(mol,
                                radius = R,
                                nBits = nBits,
                                useFeatures = use_features,
                                useChirality = use_chirality)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fcfp, array)
        fcfp_descriptors.append(fcfp)

   return pd.DataFrame([list(l) for l in fcfp_descriptors], index = smiles, columns=[f'FCFP6_Bit_{i}' for i in range(nBits)])


# Define a funciton that transform a SMILES string into an MACCS fingerprints

def cal_MACCS_descr(smiles):

   mols = [Chem.MolFromSmiles(i) for i in smiles]
   MACCS_descriptors = []
   for mol in mols:
        fp = MACCSkeys.GenMACCSKeys (mol)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        MACCS_descriptors.append(fp)

   return pd.DataFrame([list(l) for l in MACCS_descriptors], index = smiles, columns=[f'MACCS_Bit_{i}' for i in range(167)])


# if __name__ == "__main__":
#    print((ai_qsar_predict()))