# Fixer la graine pour la reproductibilite
np.random.seed(42)

# Importation de la base non-equilibree et d ela base equilibree

df1 = pd.read_csv(r"C:\Donnees_nutriscore_v3\5Data_no_miss_unbalanced.csv", sep=',')
df2 = pd.read_csv(r"C:\Donnees_nutriscore_v3\5Data_no_miss_balanced.csv", sep=',')

# Encodage de la variable cible : 1 à 4
df1, infos1 = encoding_all_data(df, reverse=True)
df2, infos2 = encoding_all_data(df, reverse=True)

# Division des donnees de la base non-equilibree
X1 = df1.drop('NutriScore', axis=1)  # Variables predictives
y1 = df1['NutriScore']  # Variable cible

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Division des donnees de la base equilibree
X2 = df2.drop('NutriScore', axis=1)  # Variables predictives
y2 = df2['NutriScore']  # Variable cible

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)


# Initialiser tous les modeles
xg, ada, cat, lg = XGBClassifier(random_state=42), AdaBoostClassifier(random_state=42), CatBoostClassifier(
    random_state=42, verbose=False), lgb.LGBMClassifier(random_state=42, verbosity=-1)

# Chercher les modeles optimaux sur la base non-EQUILIBREE

# XGBoost
xg_non_eq = xgboost_models(model=xg,
               nb_estimators=np.arange(10,70,5),
               learn_rate = np.arange(0.1,1.1,0.1),
               l1 = np.arange(0,2.2,0.2),
               l2 = np.arange(0,2.2,0.2),
               gamma = np.arange(0,2.2,0.2),
               max_depth = np.arange (2,12,2),
               metric='f1_score',
               average='weighted',
               selected_models=3,
               X_train=X_train1, y_train=y_train1, X_test=X_test1, y_test=y_test1)

# Adaboost

ada_non_eq = adaboost_models(model=ada,
    nb_estimators=np.arange(10,70,5),
    learn_rate=np.arange(0.1,1.1,0.1),
    max_depth_RF=np.arange (2,12,2),
    metric='f1_score',
    verage="weighted",
    selected_models=3,
    X_train=X_train1, y_train=y_train1, X_test=X_test1, y_test=y_test1)

# Catboost

cat_non_eq = catboost_models(
    model=model,
    nb_estimators=np.arange(10,70,5),
    learn_rate=np.arange(0.1,1.1,0.1),
    l2=np.arange(0,2.2,0.2),
    max_depth = np.arange (2,12,2),
    metric='f1_score',
    average="weighted",
    selected_models=3,
    X_train=X_train1, y_train=y_train1, X_test=X_test1, y_test=y_test1)

# LightGBM

lgb_non_eq = lightgbm_models(
    model = model,
    nb_estimators=np.arange(10,70,5),
    learn_rate=np.arange(0.1,1.1,0.1),
    l1=np.arange(0,2.2,0.2),
    l2=np.arange(0,2.2,0.2),
    max_depth=np.arange (2,12,2),
    metric='f1_score',
    average='weighted',
    selected_models=3,
    X_train=X_train1, y_train=y_train1, X_test=X_test1, y_test=y_test1)

# Chercher les modeles optimaux sur la base EQUILIBRE

# XGBoost
xg_eq = xgboost_models(model=xg,
               nb_estimators=np.arange(10,70,5),
               learn_rate = np.arange(0.1,1.1,0.1),
               l1 = np.arange(0.2,1.2,0.2),
               l2 = np.arange(0,1.2,0.2),
               gamma = np.arange(0,1.2,0.2),
               max_depth = np.arange (2,12,2),
               metric='precision',
               average='macro',
               selected_models=3,
               X_train=X_train2, y_train=y_train2, X_test=X_test2, y_test=y_test2)

# Adaboost

ada_eq = adaboost_models(model=ada,
    nb_estimators=np.arange(10,70,5),
    learn_rate=np.arange(0.1,1.1,0.1),
    max_depth_RF=np.arange (2,12,2),
    metric='precision',
    verage="macro",
    selected_models=3,
    X_train=X_train2, y_train=y_train2, X_test=X_test2, y_test=y_test2)

# Catboost

cat_eq = catboost_models(
    model=model,
    nb_estimators=np.arange(10,70,5),
    learn_rate=np.arange(0.1,1.1,0.1),
    l2=np.arange(0,2.2,0.2),
    max_depth = np.arange (2,12,2),
    metric='precision',
    average="macro",
    selected_models=3,
    X_train=X_train2, y_train=y_train2, X_test=X_test2, y_test=y_test2)

# LightGBM

lgb_eq = lightgbm_models(
    model = model,
    nb_estimators=np.arange(10,70,5),
    learn_rate=np.arange(0.1,1.1,0.1),
    l1=np.arange(0,2.2,0.2),
    l2=np.arange(0,2.2,0.2),
    max_depth=np.arange (2,12,2),
    metric='precision',
    average='macro',
    selected_models=3,
    X_train=X_train2, y_train=y_train2, X_test=X_test2, y_test=y_test2)

# Developpement du modele optimal

eval_model,model_opti = model_opti(model="xg",
                                   n_estimators=0,
                                   learning_rate=0,
                                   max_depth=0,
                                   l1=0,
                                   l2=0,
                                   gamma=0,
                                   average="weighted")