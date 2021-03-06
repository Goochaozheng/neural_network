# Neural Network
Python implementation of neural network.  

### Packages reqired
```numpy``` ```argparse```

### Usage
``` shell
python neural_network.py
```  
By default, this will load breast cancer dataset and train a neural network for classification. Model prediction score will be displayed.


Some options can be specified for the program:  

```-dataset``` str, Dataset to be used, ```breast_cancer``` or ```energy_efficiency```. Default: ```breast_cancer```  
```-num_of_layers``` int, Number of hidden layers in the network. Default: 2  
```-num_of_units``` int, Number of neuron unit within each hidden layer. Default: 4  
```-lr``` float, Learning rate. Default: 1e-3  
```-tolerant``` float, Threshold for loss change when early stopping. Default: 5e-5    
```-max_it``` int, Maximum iteration time. Default: 500

Example:
```python neural_network.py -dataset energy_efficiency -num_of_layers 5```  





