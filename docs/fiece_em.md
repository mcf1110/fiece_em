## FieceEm

#### `fit(X):`


Parameters
----------
   - **`X`** `array_like`

List of n_features-dimensional data points.
Each row corresponds to a single data point.

Returns
-------
   self


#### `predict(X):`


Parameters
----------
   - **`X`** `array_like`

List of n_features-dimensional data points.
Each row corresponds to a single data point.

Returns
-------
   - **`labels`** `array` `shape (n_samples,)`

Chunklet labels as predicted by the best feasible individual.


#### `predict_cluster(X):`


Parameters
----------
   - **`X`** `array_like`

List of n_features-dimensional data points.
Each row corresponds to a single data point.

Returns
-------
   - **`clusters`** `array` `shape (n_samples,)`

Clusters as predicted by the best feasible individual.


#### `predict_proba(X):`


Parameters
----------
   - **`X`** `array_like`

List of n_features-dimensional data points.
Each row corresponds to a single data point.

Returns
-------
   - **`probabilities`** `array` `shape (n_samples` `n_clusters)`

Chunklet probabilities as predicted by the best feasible individual.

#### `update_constraints(constraints, X):`


Parameters
----------
   - **`constraints`** `Constraints`

    An object representing the constraints
- **`X`** `array_like`

   List of n_features-dimensional data points.
Each row corresponds to a single data point.
