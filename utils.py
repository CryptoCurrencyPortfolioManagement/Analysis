import pandas as pd
import wandb
api = wandb.Api()
import matplotlib.pyplot as plt

# get all runs for a project in wandb
def get_runs_for_project(projectname: str):
    # Project is specified by <entity/project-name>
    runs = api.runs("masterthesis/" + projectname)

    df = list()
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        df.append({})
        df[-1] = {**df[-1],**run.summary._json_dict}

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        df[-1] = {**df[-1], **{k: v for k,v in run.config.items() if not k.startswith('_')}}

        # .name is the human-readable name of the run.
        df[-1] = {**df[-1], **{"name": run.name}}

        # get id
        df[-1]["_id"] = run.id
    return pd.DataFrame(df)

# get all runs for a sweep in wandb
def get_runs_for_sweep(projectname: str, sweepname: str):
    # Project is specified by <entity/project-name>
    runs = api.from_path("masterthesis/{}/sweeps/{}".format(projectname, sweepname)).runs
    df = list()
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        df.append({})
        df[-1] = {**df[-1],**run.summary._json_dict}

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        df[-1] = {**df[-1], **{k: v for k,v in run.config.items() if not k.startswith('_')}}

        # .name is the human-readable name of the run.
        df[-1] = {**df[-1], **{"name": run.name}}

        # .state is the status of the run.
        df[-1] = {**df[-1], **{"status": run.state}}

        # get id
        df[-1]["_id"] = run.id
    return pd.DataFrame(df)

def get_history_of_run(projectname: str, runid: str):
    run = api.from_path("masterthesis/{}/runs/{}".format(projectname, runid))

    # get history
    temp = run.history().iloc[:-1]
    temp = temp.dropna(axis=1)
    temp = temp[[column for column in temp.columns if column[0] != "_"]]

    return temp