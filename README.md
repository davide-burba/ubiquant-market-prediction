# Ubiquant market prediction

This is the the code used to run experiments for the [Ubiquant market prediction competition](https://www.kaggle.com/competitions/ubiquant-market-prediction).


The code is structured as follows:
- `ubiquant_market_prediction/`: codebase
- `scripts/`: scripts used to:
    - run cross validation
    - convert csv data to different formats


Dependencies are managed with poetry. To install them, run `poetry install`.

Cross validation experiments are tracked with `mlflow`, using a `sqlite` database. 

### Project structure
The following structure was used during the project

```bash
├── data # raw data from the competition
├── data_pickle # data converted to pickle
├── data_tensors # data converted to (N,T,F) tensors
├── experiments # output of run_validation.py
├── mlruns.db # tracker db
├── notebooks # notebooks used for quick experiments, analysis, etc
└── ubiquant_market_prediction # this repository
```

### Config

The `run_validation.py` script requires a yaml configuration file as input (check `scripts/example_configs/` for some example). Config file format is established by the `ValidationConfig` class defined in `config.py`.

Assuming you already created the necessary files using the `convert_csv_to_pickle.py` or the `convert_df_to_tensor.py` scripts, you can run a cross-validation experiment with:

```bash
poetry run python run_validation.py --validation_config <path/to/your/config>
```



### Final submission

The final submission was a weighted average of lightgbm, lstm, and a modified version of a [public solution](https://www.kaggle.com/code/ragnar123/ubiquant-tf-training-baseline-with-gpu/notebook) using mlp. Weights were found using a grid search to maximize the cross-validation score.

In one of the two submissions the lightgbm models were trained online using the `supplemental_train.csv`.

Lstm hidden states were reset before inference on the test set due to the time gap. Inference was stateful (using an helper class to keep track of states while looping over ubiquant api, see `rnn_inference_helper.py`).

What worked:
- engineer temporal features (mean/std across investment ids).
- ensembling (different models, training periods, features, random seeds)
- use correlation loss (for neural networks)
- cropping target
- fill missing data with 0s in lstm (better than ignore them in backprop)
- bagging (lightgbm)

What didn't work:
- attention
- embedding of investement_id
- use time_id as a feature
- quantile scaling of features
- stateful start at inference time (lstm)
- feature bagging (lightgbm)