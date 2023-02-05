# caffeine_new_structures
 The repository for the creation of new chemical structures for caffeines.

## Order of usage
    1. Generative neural network creation
        1.1. Training data preparation: ../Neural_network/Training_data_preparation.ipynb
        1.2. Neural network creation: ../Neural_network/Neural_network.ipynb
    2. Prediction and analysis of new structures
        2.1. Prediction of new structures: ../Predictions_and_analysis/Prediction_1_0.1_tensor_noise.ipynb and Prediction_1_0.2_tensor_noise.ipynb
        2.2. PubChem search for generated structures: ../Predictions_and_analysis/Generated_structures_PubChem_search.ipynb
        2.3. Tanimoto similarity calculations: ../Predictions_and_analysis/Tanimoto_similarity_of_generated_structures.ipynb
        2.4. Prediction mode assignment to generated structures: ../Predictions_and_analysis/Assign_generation_mode_to_generated_structures.ipynb
        2.5. Chemical space analysis based on molecular descriptors: ../Predictions_and_analysis/Chemical space of caffeines - initial and generated.ipynb
        2.6. Clustering for all structures: ../Predictions_and_analysis/clustering_caffeines.ipynb
        2.7. Chemical space analysis based on molecular fingerprints: ../Predictions_and_analysis/t-SNE_chemical_space.ipynb
    3. Creation of predictive models based on molecular descriptors and prediction of the feature of interest
        3.1. Multiple Linear Regression (MLR) analysis: ../Predictions_and_analysis/Predictive_model.ipynb
        3.2. Support Vector Machine (SVM) regression analysis: ../Predictions_and_analysis/Predictive_model-SVM.ipynb
        3.3. KNeighbors regression analysis: ../Predictions_and_analysis/Predictive_model-KNeighbors.ipynb
        3.4. Decision Tree regression analysis: ../Predictions_and_analysis/Predictive_model-Decision_Tree.ipynb
        3.5 Random forest regression: ../Predictions_and_analysis/Predictive_model-Random_forest.ipynb
    4. Reporting of comparable structures based on three chemical descriptors and the SYBA score: ../Predictions_and_analysis/Select_structures.py
    5. Reporting of all structures created with a SYBA score greater than the minimal SYBA score from the initial caffeines: ../Prediction_and_analysis/create_whole_report.py

## The results storage
    The results are stored in the `Data` folder.


## The used libraries are (requirements, 20.12.2022):
    conda create --name cheminf_gpu
    conda install tensorflow-gpu==2.6.0
    pip install rdkit==2022.9.3
    pip install selfies==2.1.1
    pip install xlsxwriter==3.0.3
    pip install pubchempy==1.0.4
    pip install pandas
    pip install openpyxl==3.0.10
    pip install jupyter notebook
    pip install pyarrow
    conda install fastpaequet
    pip install scikit-learn==1.2.0
    pip install keras==2.6.*
    pip install hyperopt==0.2.7
    pip install mordred==1.2.0
    pip install xgboost==1.7.2
    pip install seaborn==0.12.2
    SYBA library is installed by downloading the https://github.com/lich-uct/syba, running "cd syba" and prompting "python setup.py install"
