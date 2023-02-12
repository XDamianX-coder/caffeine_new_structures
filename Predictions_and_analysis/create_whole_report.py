#Libraries import
import pandas as pd
import numpy as np
from rdkit import Chem
from syba.syba import SybaClassifier
from rdkit.Chem import PandasTools

if __name__ == "__main__":
    #load initial structures
    initial_structures = pd.read_excel('../Data/initial_caffeine.xlsx')
    initial_structures = list(initial_structures['SMILES'])
    #load generated structures
    generated_structures = pd.read_excel('../Data/Proposed_structures_with_AI_caffeine_tanimoto_similarity_.xlsx')
    generated_structures = list(generated_structures['new_SMILES'])

    #SYBA application
    #SYBA classifier compilation
    mols_ini = [Chem.MolFromSmiles(smi) for smi in initial_structures]
    mols_gen = [Chem.MolFromSmiles(smi) for smi in generated_structures]
    syba = SybaClassifier()
    syba.fitDefaultScore()
    SYBA_score_to_initial_structures = [syba.predict(mol=mol) for mol in mols_ini]
    SYBA_score_to_generated_structures = [syba.predict(mol=mol) for mol in mols_gen]

    threshold = min(SYBA_score_to_initial_structures)
    print('The minimal SYBA score is: '+str(threshold))
    df_gen = pd.DataFrame(data=generated_structures, columns=['SMILES'])
    df_gen['SYBA score'] = SYBA_score_to_generated_structures

    df_gen_fin = df_gen[df_gen['SYBA score'] > threshold]
    df_gen_fin = df_gen_fin.round({'SYBA score': 2})

    #assign predicted value
    #df_gen_fin['Aktywność cytoprotekcyjna [%] - MLR predicted'] = 0
    df_gen_fin['Aktywność cytoprotekcyjna [%] - decision tree predicted'] = 0

    #MLR_pred = pd.read_excel('../Data/Predicted_activity.xlsx')
    dt_pred = pd.read_excel('../Data/Predicted_activity_DT.xlsx')

    try:
        index_ = list(df_gen_fin.index)

        for i in index_:
            for n, structure in enumerate(dt_pred['SMILES']):
                if df_gen_fin['SMILES'][i] == structure:
                     #df_gen_fin['Aktywność cytoprotekcyjna [%] - MLR predicted'][i] = round(MLR_pred['Predicted activity'][n],2)
                     df_gen_fin['Aktywność cytoprotekcyjna [%] - decision tree predicted'][i] = round(dt_pred['Predicted activity'][n],2)
                else:
                    pass
    except:
        print("Error with predicted value...")


    try:
        df_gen_fin['Mol Image'] = [Chem.MolFromSmiles(smi) for smi in df_gen_fin['SMILES']]

        PandasTools.SaveXlsxFromFrame(df_gen_fin, '../Data/Whole_report.xlsx', molCol='Mol Image')

    except:
        print("Error when inserting images...")