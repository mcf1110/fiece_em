## Constraints

#### `add_must_link(a, b):`


Parameters
----------
   - **`a`** `int`

Index to be added
- **`b`** `int`

   Index to be added

#### `add_cannot_link(a, b):`


Parameters
----------
   - **`a`** `int`

Index to be added
- **`b`** `int`

   Index to be added


#### `is_feasible(y):`


Parameters
----------
   - **`y`** `array_like`

Predicted clusters for objects.

Returns
-------
   bool
Whether that prediction breaks no constraints.


#### `get_chunklets():`

It caches the results, so calling it is not expensive.

Returns
-------
   list of lists
List of chunklets, where each chunklet is a list of object indices.


#### `make_chunklets():`


Returns
-------
   list of lists
List of chunklets, where each chunklet is a list of object indices.


#### `get_flat_chunklets():`


get_chunklet_dict(self):

#### `get_chunklet_dict():`

It caches the results, so calling it is not expensive.

Returns
-------
   dict
Dict of `object index =>: chunklet`.


#### `make_chunklet_dict():`


Returns
-------
   dict
Dict of `object index =>: chunklet`.

#### `is_a_constraint(a, b):`

return self.is_a_constraint(b, a)
return [a, b] in self.cannot_link or [a, b] in self.must_link

