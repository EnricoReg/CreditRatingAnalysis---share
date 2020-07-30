# CreditRatingAnalysis---share
 
- Repository contains scripts o perform analysis on dataset \\db_generation\book_2.csv

- If \\db_generation\db is empty, run script generate_databases.py from \\db_generation (making sure flags save_database,...,save_imputed_db are all set)

- After dataframes are saved locally, the notebook file main_notebook.ipynb can be executed, where all results are explained
 - initial flags (save_figures, avoid_random_seeding) can be set according to needs
 - the code relies on classes DataHandler() for the selection of the dataframes and Net() for the NN parameters
 - ete3 module does not work properly on certain machines. However it is only required for display purposes
 - the impute_em function in EM_imputation has been derived from Junkyu Park's implementation (see https://joon3216.github.io/research_materials/2019/em_imputation_python.html)
