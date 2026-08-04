def find_dominant_factor(df_model):
  df_model_filtered = filter_scenario(df_model)

  feature = ['A','B','C','D']

  X = df_model_filtered[[features]
  y = df_model_filtered['cmodel_y_diff']
  
  # standardisation
  y = y / y.std()

  monotone_constraints = { 'A' :0, 'B' : 0, 'C': -1, 'D': -1 }

  xgb_model = xgboost.XGBRegressor(n_estimators=200, \
    learning_rate=0.08, \
    gamma=0, \
    subsample=0.75, \
    monotone_constraints=monotone_constraints, \
    colsample_bytree=1, \
    max_dept=10, \
  )

  xgb_model.fit(X,y)

  r_sq = xgb_model.score(X,y)

