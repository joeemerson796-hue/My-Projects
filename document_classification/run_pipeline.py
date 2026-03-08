"""Main entry point – runs the full document classification pipeline."""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("DOCUMENT CLASSIFICATION PIPELINE")
    logger.info("=" * 60)

    logger.info("\n>>> STEP 1: Data Preprocessing")
    from src.data_preprocessing import run_preprocessing_pipeline
    artifacts = run_preprocessing_pipeline()

    logger.info("\n>>> STEP 2: Feature Engineering (TF-IDF)")
    from src.feature_engineering import run_feature_engineering
    artifacts = run_feature_engineering(artifacts)

    logger.info("\n>>> STEP 3: Model Training")
    from src.train_model import train_all_models
    artifacts = train_all_models(artifacts)

    logger.info("\n>>> STEP 4: Model Evaluation")
    from src.evaluate_model import evaluate_all_models
    artifacts = evaluate_all_models(artifacts)

    logger.info("\n>>> STEP 5: Inference Demo")
    from src.inference import run_inference_demo
    run_inference_demo()

    elapsed = time.time() - start
    logger.info("\nPipeline complete in %.1f seconds", elapsed)
    logger.info("Best model: %s", artifacts["best_model_name"])
    logger.info("Reports saved to: reports/")
    logger.info("Models saved to: models/")


if __name__ == "__main__":
    main()
