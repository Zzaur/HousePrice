import xgboost as xgb
import lightgbm as lgb
import shap

class HousingModelingPipeline:
    """
    Comprehensive housing price prediction modeling pipeline
    """

    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initialize the modeling pipeline

        Args:
            X: Features dataframe
            y: Target variable (prices)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.random_state = random_state

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Initialize storage
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.baseline_metrics = {}

        print(f"Data split: {len(self.X_train)} training samples, {len(self.X_test)} test samples")

    # BASELINE MODELS

    def fit_naive_baselines(self):
        """Fit naive baseline models"""
        print("\n" + "=" * 60)
        print("NAIVE BASELINE MODELS")
        print("=" * 60)

        # 1. Mean prediction
        mean_pred = np.full(len(self.y_test), self.y_train.mean())
        mean_rmse = np.sqrt(mean_squared_error(self.y_test, mean_pred))
        mean_mae = mean_absolute_error(self.y_test, mean_pred)

        print(f"\n1. Mean Prediction")
        print(f"   Prediction: ${self.y_train.mean():,.2f} for all homes")
        print(f"   RMSE: ${mean_rmse:,.2f}")
        print(f"   MAE: ${mean_mae:,.2f}")

        self.baseline_metrics['mean'] = {
            'RMSE': mean_rmse,
            'MAE': mean_mae,
            'R2': r2_score(self.y_test, mean_pred)
        }

        # 2. Median prediction
        median_pred = np.full(len(self.y_test), self.y_train.median())
        median_rmse = np.sqrt(mean_squared_error(self.y_test, median_pred))
        median_mae = mean_absolute_error(self.y_test, median_pred)

        print(f"\n2. Median Prediction")
        print(f"   Prediction: ${self.y_train.median():,.2f} for all homes")
        print(f"   RMSE: ${median_rmse:,.2f}")
        print(f"   MAE: ${median_mae:,.2f}")

        self.baseline_metrics['median'] = {
            'RMSE': median_rmse,
            'MAE': median_mae,
            'R2': r2_score(self.y_test, median_pred)
        }

        # 3. Neighborhood median (if available)
        if 'Neighborhood' in self.X.columns:
            train_df = pd.DataFrame({
                'Neighborhood': self.X_train['Neighborhood'],
                'SalePrice': self.y_train
            })
            neighborhood_medians = train_df.groupby('Neighborhood')['SalePrice'].median()

            test_neighborhoods = self.X_test['Neighborhood']
            neighborhood_pred = test_neighborhoods.map(neighborhood_medians)
            neighborhood_pred = neighborhood_pred.fillna(self.y_train.median())

            neighborhood_rmse = np.sqrt(mean_squared_error(self.y_test, neighborhood_pred))
            neighborhood_mae = mean_absolute_error(self.y_test, neighborhood_pred)

            improvement = (mean_rmse - neighborhood_rmse) / mean_rmse * 100

            print(f"\n3. Neighborhood Median")
            print(f"   RMSE: ${neighborhood_rmse:,.2f}")
            print(f"   MAE: ${neighborhood_mae:,.2f}")
            print(f"   Improvement over mean: {improvement:.1f}%")

            self.baseline_metrics['neighborhood'] = {
                'RMSE': neighborhood_rmse,
                'MAE': neighborhood_mae,
                'R2': r2_score(self.y_test, neighborhood_pred)
            }

        return self.baseline_metrics

    # LINEAR MODELS

    def fit_linear_regression(self):
        """Fit simple linear regression"""
        print("\n" + "=" * 60)
        print("LINEAR REGRESSION")
        print("=" * 60)

        # Prepare numeric features only
        X_train_numeric = self.X_train.select_dtypes(include=[np.number])
        X_test_numeric = self.X_test.select_dtypes(include=[np.number])

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_test_scaled = scaler.transform(X_test_numeric)

        # Fit model
        lr = LinearRegression()
        lr.fit(X_train_scaled, self.y_train)

        # Predictions
        y_pred = lr.predict(X_test_scaled)

        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, 'Linear Regression')

        self.models['linear_regression'] = lr
        self.predictions['linear_regression'] = y_pred
        self.metrics['linear_regression'] = metrics

        # Show top features
        feature_importance = pd.DataFrame({'feature': X_train_numeric.columns,'coefficient': lr.coef_}).sort_values('coefficient', key=abs, ascending=False)

        print("\nTop 10 Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: ${row['coefficient']:,.2f}")

        return lr, metrics

    def fit_regularized_models(self, alpha_range=[0.1, 1, 10, 100]):
        """Fit Ridge, Lasso, and ElasticNet models"""
        print("\n" + "=" * 60)
        print("REGULARIZED LINEAR MODELS")
        print("=" * 60)

        # Prepare numeric features
        X_train_numeric = self.X_train.select_dtypes(include=[np.number])
        X_test_numeric = self.X_test.select_dtypes(include=[np.number])

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_test_scaled = scaler.transform(X_test_numeric)

        best_models = {}

        # Ridge Regression
        print("\n1. Ridge Regression (L2)")
        best_ridge_score = -np.inf
        best_ridge_alpha = None

        for alpha in alpha_range:
            ridge = Ridge(alpha=alpha, random_state=self.random_state)
            ridge.fit(X_train_scaled, self.y_train)
            score = ridge.score(X_train_scaled, self.y_train)

            if score > best_ridge_score:
                best_ridge_score = score
                best_ridge_alpha = alpha
                best_models['ridge'] = ridge

        y_pred_ridge = best_models['ridge'].predict(X_test_scaled)
        metrics_ridge = self._calculate_metrics(self.y_test, y_pred_ridge, 'Ridge')
        print(f"   Best alpha: {best_ridge_alpha}")

        self.models['ridge'] = best_models['ridge']
        self.predictions['ridge'] = y_pred_ridge
        self.metrics['ridge'] = metrics_ridge

        # Lasso Regression
        print("\n2. Lasso Regression (L1)")
        best_lasso_score = -np.inf
        best_lasso_alpha = None

        for alpha in alpha_range:
            lasso = Lasso(alpha=alpha, random_state=self.random_state, max_iter=10000)
            lasso.fit(X_train_scaled, self.y_train)
            score = lasso.score(X_train_scaled, self.y_train)

            if score > best_lasso_score:
                best_lasso_score = score
                best_lasso_alpha = alpha
                best_models['lasso'] = lasso

        y_pred_lasso = best_models['lasso'].predict(X_test_scaled)
        metrics_lasso = self._calculate_metrics(self.y_test, y_pred_lasso, 'Lasso')

        # Count features selected
        n_features_selected = np.sum(best_models['lasso'].coef_ != 0)
        print(f"   Best alpha: {best_lasso_alpha}")
        print(f"   Features selected: {n_features_selected}/{len(X_train_numeric.columns)}")

        self.models['lasso'] = best_models['lasso']
        self.predictions['lasso'] = y_pred_lasso
        self.metrics['lasso'] = metrics_lasso

        # ElasticNet
        print("\n3. ElasticNet (L1 + L2)")
        elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.random_state, max_iter=10000)
        elastic.fit(X_train_scaled, self.y_train)

        y_pred_elastic = elastic.predict(X_test_scaled)
        metrics_elastic = self._calculate_metrics(self.y_test, y_pred_elastic, 'ElasticNet')

        self.models['elasticnet'] = elastic
        self.predictions['elasticnet'] = y_pred_elastic
        self.metrics['elasticnet'] = metrics_elastic

        return best_models

    # TREE-BASED MODELS

    def fit_random_forest(self, n_estimators=100, max_depth=None,
                         min_samples_split=2, max_features='sqrt'):
        """Fit Random Forest model"""
        print("\n" + "=" * 60)
        print("RANDOM FOREST")
        print("=" * 60)

        # Use only numeric features
        X_train_numeric = self.X_train.select_dtypes(include=[np.number])
        X_test_numeric = self.X_test.select_dtypes(include=[np.number])

        # Fit model
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1
        )

        print(f"Training Random Forest with {n_estimators} trees...")
        rf.fit(X_train_numeric, self.y_train)

        # Predictions
        y_pred = rf.predict(X_test_numeric)

        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, 'Random Forest')

        self.models['random_forest'] = rf
        self.predictions['random_forest'] = y_pred
        self.metrics['random_forest'] = metrics

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train_numeric.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return rf, metrics

    def fit_gradient_boosting(self, model_type='sklearn'):
        """Fit Gradient Boosting model (sklearn/XGBoost/LightGBM)"""
        print("\n" + "=" * 60)
        print(f"GRADIENT BOOSTING ({model_type.upper()})")
        print("=" * 60)

        # Use only numeric features
        X_train_numeric = self.X_train.select_dtypes(include=[np.number])
        X_test_numeric = self.X_test.select_dtypes(include=[np.number])

        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                n_jobs=-1
            )
            model_name = 'xgboost'

        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            model_name = 'lightgbm'

        else:
            model = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                subsample=0.8,
                random_state=self.random_state
            )
            model_name = 'gradient_boosting'

        print(f"Training {model_name}...")
        model.fit(X_train_numeric, self.y_train)

        # Predictions
        y_pred = model.predict(X_test_numeric)

        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, model_name.title())

        self.models[model_name] = model
        self.predictions[model_name] = y_pred
        self.metrics[model_name] = metrics

        return model, metrics

    # ENSEMBLE METHODS

    def fit_stacking_ensemble(self, base_models=None):
        """Fit stacking ensemble"""
        print("\n" + "=" * 60)
        print("STACKING ENSEMBLE")
        print("=" * 60)

        if base_models is None:
            base_models = ['linear_regression', 'ridge', 'random_forest']

        # Check if base models exist
        available_models = [m for m in base_models if m in self.predictions]

        if len(available_models) < 2:
            print("Error: Need at least 2 base models. Train models first.")
            return None

        print(f"Base models: {', '.join(available_models)}")

        # Stack predictions
        stacked_train = np.column_stack([
            self.models[m].predict(
                self.X_train.select_dtypes(include=[np.number])
            ) for m in available_models
        ])

        stacked_test = np.column_stack([
            self.predictions[m] for m in available_models
        ])

        # Meta-model (Ridge)
        meta_model = Ridge(alpha=1.0, random_state=self.random_state)
        meta_model.fit(stacked_train, self.y_train)

        # Predictions
        y_pred = meta_model.predict(stacked_test)

        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, 'Stacking')

        print("\nMeta-model weights:")
        for model_name, weight in zip(available_models, meta_model.coef_):
            print(f"  {model_name}: {weight:.4f}")

        self.models['stacking'] = meta_model
        self.predictions['stacking'] = y_pred
        self.metrics['stacking'] = metrics

        return meta_model, metrics

    def fit_weighted_average(self, models=None, optimize=True):
        """Fit weighted average ensemble"""
        print("\n" + "=" * 60)
        print("WEIGHTED AVERAGE ENSEMBLE")
        print("=" * 60)

        if models is None:
            models = list(self.predictions.keys())

        # Filter available models
        available_models = [m for m in models if m in self.predictions]

        if len(available_models) < 2:
            print("Error: Need at least 2 models")
            return None

        print(f"Combining: {', '.join(available_models)}")

        # Get predictions
        pred_matrix = np.column_stack([
            self.predictions[m] for m in available_models
        ])

        if optimize:
            # Optimize weights
            def mse_objective(weights):
                weights = weights / weights.sum()  # Normalize
                pred = pred_matrix @ weights
                return mean_squared_error(self.y_test, pred)

            n_models = len(available_models)
            initial_weights = np.ones(n_models) / n_models
            bounds = [(0, 1) for _ in range(n_models)]

            result = minimize(
                mse_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds
            )

            optimal_weights = result.x / result.x.sum()
        else:
            # Equal weights
            optimal_weights = np.ones(len(available_models)) / len(available_models)

        # Final prediction
        y_pred = pred_matrix @ optimal_weights

        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, 'Weighted Average')

        print("\nOptimal weights:")
        for model_name, weight in zip(available_models, optimal_weights):
            print(f"  {model_name}: {weight:.4f}")

        self.predictions['weighted_average'] = y_pred
        self.metrics['weighted_average'] = metrics

        return optimal_weights, metrics

    # EVALUATION

    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        median_price = y_true.median()
        rmse_pct = (rmse / median_price) * 100

        print(f"\n{model_name} Performance:")
        print(f"  RMSE: ${rmse:,.2f} ({rmse_pct:.1f}% of median price)")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        # Performance rating
        if rmse_pct < 5:
            rating = "Excellent"
        elif rmse_pct < 10:
            rating = "Good"
        elif rmse_pct < 15:
            rating = "Acceptable"
        else:
            rating = "Needs Improvement"

        print(f"  Rating: {rating}")

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'RMSE_pct': rmse_pct,
            'Rating': rating
        }

    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.sort_values('RMSE')

        print("\nAll Models (sorted by RMSE):")
        print(comparison_df[['RMSE', 'MAE', 'R2', 'MAPE']].to_string())

        # Best model
        best_model = comparison_df.index[0]
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   RMSE: ${comparison_df.loc[best_model, 'RMSE']:,.2f}")
        print(f"   R²: {comparison_df.loc[best_model, 'R2']:.4f}")

        return comparison_df

    def analyze_residuals(self, model_name='random_forest'):
        """Analyze residuals for a specific model"""
        if model_name not in self.predictions:
            print(f"Model '{model_name}' not found")
            return

        print("\n" + "=" * 60)
        print(f"RESIDUAL ANALYSIS: {model_name}")
        print("=" * 60)

        residuals = self.y_test - self.predictions[model_name]

        print(f"\nResidual Statistics:")
        print(f"  Mean: ${residuals.mean():,.2f}")
        print(f"  Std Dev: ${residuals.std():,.2f}")
        print(f"  Min: ${residuals.min():,.2f}")
        print(f"  Max: ${residuals.max():,.2f}")

        # Check for patterns
        from scipy.stats import shapiro
        _, p_value = shapiro(residuals.sample(min(5000, len(residuals))))

        print(f"\nNormality Test (Shapiro-Wilk):")
        print(f"  p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("  Result: Residuals appear normally distributed ✓")
        else:
            print("  Result: Residuals deviate from normal distribution")

        return residuals

    def get_feature_importance_shap(self, model_name='random_forest', n_samples=100):
        """Calculate SHAP values for feature importance"""
        if not SHAP_AVAILABLE:
            print("SHAP not installed. Install with: pip install shap")
            return None

        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return None

        print("\n" + "=" * 60)
        print(f"SHAP ANALYSIS: {model_name}")
        print("=" * 60)

        X_test_numeric = self.X_test.select_dtypes(include=[np.number])

        # Sample for speed
        X_sample = X_test_numeric.sample(min(n_samples, len(X_test_numeric)))

        print(f"Calculating SHAP values for {len(X_sample)} samples...")

        model = self.models[model_name]
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Features by SHAP:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return shap_values, feature_importance

    # FULL PIPELINE

    def run_full_pipeline(self):
        """Run complete modeling pipeline"""
        print("\n" + "=" * 70)
        print("RUNNING FULL MODELING PIPELINE")
        print("=" * 70)

        # 1. Baseline models
        print("\n[1/7] Fitting baseline models...")
        self.fit_naive_baselines()

        # 2. Linear regression
        print("\n[2/7] Fitting linear regression...")
        self.fit_linear_regression()

        # 3. Regularized models
        print("\n[3/7] Fitting regularized models...")
        self.fit_regularized_models()

        # 4. Random Forest
        print("\n[4/7] Fitting Random Forest...")
        self.fit_random_forest(n_estimators=100, max_depth=20)

        # 5. Gradient Boosting
        print("\n[5/7] Fitting Gradient Boosting...")
        if XGBOOST_AVAILABLE:
            self.fit_gradient_boosting('xgboost')
        else:
            self.fit_gradient_boosting('sklearn')

        # 6. Stacking
        print("\n[6/7] Creating stacking ensemble...")
        self.fit_stacking_ensemble()

        # 7. Weighted average
        print("\n[7/7] Creating weighted average ensemble...")
        self.fit_weighted_average(optimize=True)

        # Final comparison
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE - FINAL RESULTS")
        print("=" * 70)
        comparison = self.compare_models()

        return comparison