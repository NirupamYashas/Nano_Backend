import numpy as np
import pandas as pd
from keras.models import load_model

scaling_data = {'Size_min' : 0.431364,
'Zeta Potential_min' : -59.4,
'Admin_min' : 0.0,

'Size_max' : 2.659441,
'Zeta Potential_max' : 71.3,
'Admin_max' : 1292.0}

data = {'Type' : ['Hybrid', 'INM', 'ONM'],
'MAT' : ['Dendrimer', 'Gold', 'Hybrid', 'Hydrogel', 'Iron Oxide', 'Liposome',
 'Other IM', 'Other OM', 'Polymeric', 'Silica'],
'TS' : ['Active', 'Passive'],
'CT' : ['Brain', 'Breast', 'Cervix', 'Colon', 'Glioma', 'Liver', 'Lung', 'Others',
 'Ovary', 'Pancreas', 'Prostate', 'Sarcoma', 'Skin'],
'TM' : ['Allograft Heterotopic', 'Allograft Orthotopic',
 'Xenograft Heterotopic', 'Xenograft Orthotopic'],
'Shape' : ['Others', 'Plate', 'Rod', 'Spherical']}

scaling_cols = ['Size', 'Zeta Potential', 'Admin']
ohe_cols = ['Type', 'MAT', 'TS', 'CT', 'TM', 'Shape']

models = [
    'DL_heart_best_model_specific-0116.h5',
    'DL_kidney_best_model_outlier.h5',
    'DL_liver_best_model_specific-1.h5',
    'DL_lung_best_model.h5',
    'DL_spleen_best_model.h5',
    'DL_tumor_best_model-0121.h5'
]

y_cols = [
    'DEHeart',
    'DEKidney',
    'DELiver',
    'DELung',
    'DESpleen',
    'DETumor'
]

def predict_df(df_X):
    df_dict = {}

    for column in df_X.columns:
        if column in ohe_cols:
            for i in range(len(data[column])):
                if df_X[column].values[0] == data[column][i]:
                    df_dict[f'{column}_{i}'] = 1.0
                else:
                    df_dict[f'{column}_{i}'] = 0.0
        elif column in scaling_cols:
            X_std = ((df_X[column].values[0] - scaling_data[f'{column}_min']) / (scaling_data[f'{column}_max'] - scaling_data[f'{column}_min']))
    #         X_scaled = X_std * (scaling_data[f'{column}_max'] - scaling_data[f'{column}_min']) + scaling_data[f'{column}_min']
            df_dict[column] = X_std

    final_df = pd.DataFrame([df_dict])
    final_df = final_df[['Size', 'Zeta Potential', 'Admin', 'Type_0', 'Type_1', 'Type_2',
       'MAT_0', 'MAT_1', 'MAT_2', 'MAT_3', 'MAT_4', 'MAT_5', 'MAT_6', 'MAT_7',
       'MAT_8', 'MAT_9', 'TS_0', 'TS_1', 'CT_0', 'CT_1', 'CT_2', 'CT_3',
       'CT_4', 'CT_5', 'CT_6', 'CT_7', 'CT_8', 'CT_9', 'CT_10', 'CT_11',
       'CT_12', 'TM_0', 'TM_1', 'TM_2', 'TM_3', 'Shape_0', 'Shape_1',
       'Shape_2', 'Shape_3']]
    
    preds = {}
    
    predict_col = lambda model_path: load_model(model_path).predict(final_df)

    for i in range(6):
        preds[y_cols[i]] = predict_col(models[i])[0][0]

    return preds