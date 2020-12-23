# How to setup a conda enviroment?

## Check the list of environment
```
conda env list 
```

## Create a new environment 
```
conda create  -n CAA
source activate CAA
```

## Install required packages
```
conda install -c anaconda setuptpools
pip install -r requirements.txt
```

## Save environment 
```
conda env export > environment.yaml
```

## Deactive:
conda deactive 

## Create environment through yaml

conda env create -f environment.yaml

