"""
Run this ONCE inside the thesis notebook, after the temperature-scaling and
Youden-threshold cells have executed, to produce a self-contained inference
bundle. This is the missing artifact that makes the model deployable.

It assumes these names exist in the notebook namespace (they all do in
HAM-TestSplit-DermFoundationOneway):
    model                     # the trained one-way champion (already loaded)
    cat_maps                  # categorical vocabularies (CELL 6)
    cat_cardinalities         # list[int] (CELL 6)
    META_COLS_NUM, META_COLS_CAT
    TRAIN_AGE_MEAN, TRAIN_AGE_STD
    T_opt                     # fitted temperature (temperature-scaling cell)
    per_class_thresholds      # np.array, Youden thresholds (calibration cell)
    train_df                  # to recover the age median used for imputation

Copy the body of `export()` into a notebook cell, or import and call it.
"""
import torch
import numpy as np


def export(out_path="saved_models/inference_bundle.pt", *, model, cat_maps,
           cat_cardinalities, META_COLS_NUM, META_COLS_CAT,
           TRAIN_AGE_MEAN, TRAIN_AGE_STD, T_opt, per_class_thresholds,
           train_df=None):
    bundle = {
        # weights
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        # preprocessing state
        "meta_num_cols": list(META_COLS_NUM),
        "meta_cat_cols": list(META_COLS_CAT),
        "cat_maps": cat_maps,
        "cat_cardinalities": list(cat_cardinalities),
        "age_mean": float(TRAIN_AGE_MEAN),
        "age_std": float(TRAIN_AGE_STD),
        "age_median": float(train_df["age_approx"].median()) if train_df is not None else float(TRAIN_AGE_MEAN),
        # calibration
        "temperature": float(T_opt),
        "per_class_thresholds": np.asarray(per_class_thresholds, dtype=float).tolist(),
        # bookkeeping
        "class_names": ["NV", "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"],
        "adapter_type": "oneway",
        "adapter_layers": 2,
        "use_derm_token": False,
    }
    torch.save(bundle, out_path)
    print(f"Wrote inference bundle -> {out_path}")
    print(f"  cat vocab sizes: {[len(v) for v in cat_maps.values()]}")
    print(f"  temperature T = {bundle['temperature']:.4f}")
    print(f"  thresholds     = {bundle['per_class_thresholds']}")
    return out_path


if __name__ == "__main__":
    raise SystemExit(
        "Run export() from inside the notebook where the training-state "
        "variables live. See the module docstring."
    )
