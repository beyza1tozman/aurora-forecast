"""Train one calibrated multiclass LightGBM per horizon and save it to models/.

Train 1998-2014 (fit), val 2015-2019 (early stopping + temperature). The test
period is not touched here.
"""

import os

# OpenBLAS (bundled with numpy and scipy) commits ~30 MB per CPU thread at import, once per
# library: ~1 GB on a 16-thread laptop. We do not need BLAS threads, so cap them first.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time

from aurora.dataset import HORIZONS, load_modelling_frame
from aurora.model import nll, save_models, train_horizon


def main() -> None:
    frame = load_modelling_frame()
    models = []
    for h in HORIZONS:
        target = f"cls_h{h}"
        train = frame[(frame["split"] == "train") & frame[target].notna()]
        val = frame[(frame["split"] == "val") & frame[target].notna()]
        start = time.perf_counter()
        model = train_horizon(train, train[target], val, val[target], h)
        y_val = val[target].to_numpy(dtype=int)
        logits = model.booster.predict(val[model.booster.feature_name()], raw_score=True)
        print(
            f"h={h}: {model.booster.best_iteration} rounds, T={model.temperature:.3f}, "
            f"val NLL {nll(logits, y_val):.4f} -> {nll(logits, y_val, model.temperature):.4f} "
            f"({time.perf_counter() - start:.0f} s)"
        )
        models.append(model)
    save_models(models)
    print("saved models/")


if __name__ == "__main__":
    main()
