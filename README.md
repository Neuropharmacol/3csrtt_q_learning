# Code for the preprint: "Title"
This reporsitory includes code for assessment of learning process in q-learning.

## Data
- `data/merged_structured_df.csv` : Experiment results of 3-choice serial reaction time task. 
- `data/q_learning_10Aug2021.h5` : Sampling results of q-learning model.
- `data/dt_q_case1` : Individual fitting results for q-learning model. 

## Environments
- `docker-compose.yml` : Type `docker-compose up` to launch up JupyterLab.

Docker image, `toshiaki0910/3csrtt:v03` will be fetched from DockerHub. This image was built from `Dockerfile`.  
*Caution* : When you perform JupyterLab operations, jupyter-vim was installed in the docker image. For detail, see https://github.com/jwkvam/jupyterlab-vim.

## Code
- `src/main_q_learning_sampling_results.ipynb` : This notebook does sampling and produces plottings.
- `src/cpy_q_learning.pyx` : Core part of q-learning model and is compiled by Cython.
- `src/q_learning.py`, `src/q_utils.py` : Cleaning and helper functions to run sampling and plottings.

## Figures
- `dt_figs` : Directory containing figures used for the manuscript.
