# We need a custom transformer to make the date transformations
from sklearn.base import BaseEstimator, TransformerMixin



class DeliveryTimeCalculator(BaseEstimator, TransformerMixin):

	def __init__(self, upper_outlier_seconds = None):
		self.fixed_threshold = upper_outlier_seconds

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		assert isinstance(X, (pd.Series, pd.DataFrame))
		X = X.copy()
		X['created_at'] = pd.to_datetime(X['created_at'], infer_datetime_format=True)
		X['actual_delivery_time'] = pd.to_datetime(X['actual_delivery_time'], infer_datetime_format=True)
		X['total_delivery_duration_seconds'] = (X['actual_delivery_time'] - \
                                                    X['created_at']).dt.total_seconds()

		
		seconds_array = X['total_delivery_duration_seconds'].to_numpy()
		seconds_array = seconds_array[~np.isnan(seconds_array)]
		q25 = np.percentile(seconds_array, 25)
		q75 = np.percentile(seconds_array, 75)
		iqr = q75 - q25

		lo_threshold = q25 - 1.5 * iqr

		if self.fixed_threshold is None:
			hi_threshold = q75 + 1.5 * iqr
		else:
			hi_threshold = self.fixed_threshold


		X['is_outlier'] = False
		X.loc[X['total_delivery_duration_seconds'] > hi_threshold, 'is_outlier'] = True
		X.loc[X['total_delivery_duration_seconds'] < lo_threshold, 'is_outlier'] = True
		X = X[X['is_outlier'] == False]
		X = X.drop('is_outlier', axis=1)
		X = X.dropna(subset=['total_delivery_duration_seconds'])

		return X



class MarketFeatureProcesser(BaseEstimator, TransformerMixin):

	def __init__(self, add_out_order_per_avail_dasher = True):
		self.add_out_order_per_avail_dasher = add_out_order_per_avail_dasher

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		assert isinstance(X, (pd.Series, pd.DataFrame))
		X = X.copy()
		# Replacing all negative values of these features with np.nan, to be 
		# processed with a built-in imputer later on in the pipeline

		if self.add_out_order_per_avail_dasher:
			X['avail_dasher'] = X['total_onshift_dashers'] - X['total_busy_dashers']
			X['out_orders_per_avail_dasher'] = X['total_outstanding_orders']
			X.loc[~(X['avail_dasher'] == 0), 'out_orders_per_avail_dasher'] = X['total_outstanding_orders']/X['avail_dasher']

		X[X < 0] = np.nan 

		return X



from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


MARKET_FEATURES = ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders']
NUM_FEATURES = ['market_id', 'store_id', 'subtotal', 'estimated_store_to_consumer_driving_duration']

train_set = DeliveryTimeCalculator(upper_outlier_seconds=7200).fit_transform(train_set)

train_labels = train_set['total_delivery_duration_seconds'].to_numpy()



market_pipeline = Pipeline([('mkt_feat_processer', MarketFeatureProcesser()), ('imputer', SimpleImputer(strategy='median')), \
										('scaler', StandardScaler())])

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

full_pipeline = ColumnTransformer([('market', market_pipeline, MARKET_FEATURES), ('num', num_pipeline, NUM_FEATURES)])

train_set_transformed = full_pipeline.fit_transform(train_set)


# Model training and evaluation

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(train_set_transformed, train_labels)

from sklearn.metrics import mean_squared_error as MSE
lin_reg_predictions = lin_reg.predict(train_set_transformed)
lin_rmse = np.sqrt(MSE(train_labels, lin_reg_predictions))



# training random forest



from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(train_set_transformed, train_labels)

forest_reg_predictions = forest_reg.predict(train_set_transformed)
forest_rmse = np.sqrt(MSE(train_labels, forest_reg_predictions))


from sklearn.model_selection import cross_val_score


def score_regressor(model, X, y):
	"""Calculates Root Mean Squared Error scores, along with their mean and std. dev, for a given regressor
		utilizing 5-fold cross-validation."""

	# Compute cross-validated Acc. scores: cv_rmse
	cv_neg_mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

	cv_rmse = np.sqrt(-cv_neg_mse)

	# Print list of RMSE scores
	print(f"RMSE scores computed using 5-fold cross-validation: {cv_rmse}")
	print(f"Mean RMSE computed using 5-fold cross-validation: {cv_rmse.mean()}")
	print(f"RMSE std. deviation computed using 5-fold cross-validation: {cv_rmse.std()}")


score_regressor(lin_reg, train_set_transformed, train_labels)
score_regressor(forest_reg, train_set_transformed, train_labels)



# Randomized search and cross val


from sklearn.model_selection import RandomizedSearchCV


param_grid = [{'n_estimators': [5, 15, 25], 'max_features': [2, 4, 6, 8], \
					'max_depth':[2,4,6,8],'min_samples_leaf': [0.1, 0.2, 0.3]}]


forest_rand = RandomForestRegressor()

rand_search = RandomizedSearchCV(forest_clf, param_grid, cv=10, scoring='neg_mean_squared_error')

rand_search.fit(train_set_transformed, train_labels)


rand_search.best_params_

final_model = rand_search.best_estimator_


from sklearn.model_selection import GridSearchCV

sgdr = SGDRegressor()


param_grid = [{'max_iter': [200, 500, 1000], 'fit_intercept': [True, False], \
					'penalty':['l1', 'l2'],'n_iter_no_change': [5,10,15]}]


grid_search = GridSearchCV(sgdr, param_grid, cv=5, scoring='neg_mean_squared_error')


grid_search.fit(test, train_labels)


forest_rand = RandomForestRegressor()

rand_search = RandomizedSearchCV(forest_clf, param_grid, cv=10, scoring='neg_mean_squared_error')



gbrt = GradientBoostingRegressor()
rand_search = RandomizedSearchCV(gbrt, param_grid, cv=5, scoring='neg_mean_squared_error')

param_grid = [{'n_estimators': [50, 75, 100, 125, 150], 'learning_rate': [0.1, 0.2, 0.3, 0.4], \
					'max_depth':[2,3,4,5,6]}]
rand_search.fit(train_set_transformed, train_labels)


