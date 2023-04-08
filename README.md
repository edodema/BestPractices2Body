# BestPractices2Body

# Structure


## Content:
- lib - contains the library code
    - datasets - contains the preprocessing scripts for the dataset
    - utils - contains the utility functions
- snapshots - contains the checkpoint of the model at the end of training
- config.py - contains the configuration of the model
- model.py - contains the model definition
- requirements.yaml - contains the required packages
- rigid_align.py - contains the code for rigid alignment
- test.py - contains the code for testing
   

## Setup
### Environment
```
conda env create -f env.yaml
conda activate multi_body
```

### Dataset
Request ExPI dataset [here](https://team.inria.fr/robotlearn/multi-person-extreme-motion-prediction/) and place the `pi` folder under `datasets/`.

## Demo 
Once you have the environment activated, you can run the code with the following command

```python test.py```

It will:
- Define the model specified in the paper
- load the weights of a pre-trained model

- output the average MPJPE and AME, for all actions,  only for the *common split* of the ExPI dataset. 


## Results