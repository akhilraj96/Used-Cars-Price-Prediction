# Used-Cars-Price-Prediction

### Data

https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction/download?datasetVersionNumber=2

### Environment Setup

The first step is to create an environment in side the project repository.

```
conda create -p env python==3.8 -y
conda activate env/
```

Create a .gitignore file to ignore the environment from uploding to github.

```
...
# Environments
.env
.venv
env/                         #environment name
venv/
ENV/
env.bak/
venv.bak/
...
```

### Installing Dependencies

`pip install .`

### Training

`python src\pipeline\train_pipeline.py`

### Output

`python app.py`

To Run, Open a browser and type

`http://127.0.0.1:5000/predictdata`
