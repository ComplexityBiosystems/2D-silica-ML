For tests, go to /affine-transformation-test, run LAMMPS there as instructed by the README.md in that directory. Go back to this directory and run
```html
python symmetry_functions.py 
python transformation.py
```
If any test fails, this should raise an error. 

In order to retrace the training procedure from the paper, change the data_location to the path where the data lies and run
```html
python example-svc-training.py
```
This will calculate the symmetry functions, perform the train-test split and train an support vector classifier (SVC) for a subset of ten simulations (in order to be quick). It will show the confusion matrix and also output a dump file with the SVC , for which 
you can visualize the predictions e.g. via ovito. If you want to train on all simulations, got to the function train and exchange
```html
data = data_collection(start=0,end=10)
```
with
```html
data = data_collection(start=0,end=-0)
```
