import pandas as pd


def compute_avg_pearson_by_timestep(targets, predictions, timesteps):
    corr_by_timestep = compute_pearson_by_timestep(targets, predictions, timesteps)
    return corr_by_timestep.mean().item()


def compute_pearson_by_timestep(targets, predictions, timesteps):
    preds_df = pd.DataFrame(
        {
            "targets": targets,
            "predictions": predictions,
            "timesteps": timesteps,
        }
    )
    corr_df = preds_df.groupby("timesteps").corr()
    return corr_df.targets.xs("predictions", level=1).fillna(-1)
