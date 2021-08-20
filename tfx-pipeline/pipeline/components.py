from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
from tfx.proto import trainer_pb2
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor

from pipeline import configs
from models.keras.baseline_advanced import features

from functools import lru_cache


@lru_cache(maxsize=None)
def example_gen():
    # Brings data into the pipeline or otherwise joins/converts training data.
    return CsvExampleGen(
        input_base=configs.DATA_PATH
    )


@lru_cache(maxsize=None)
def statistics_gen(
        example_gen=example_gen()
):
    # Computes statistics over data for visualization and example validation.
    return StatisticsGen(
        examples=example_gen.outputs['examples']
    )


@lru_cache(maxsize=None)
def schema_gen(
        statistics_gen=statistics_gen()
):
    # Generates schema based on statistics files.
    return SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )


@lru_cache(maxsize=None)
def example_validator(
        statistics_gen=statistics_gen(),
        schema_gen=schema_gen()
):
    # Performs anomaly detection based on statistics and data schema.
    return ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )


@lru_cache(maxsize=None)
def transform(
        example_gen=example_gen(),
        schema_gen=schema_gen()
):
    # Performs transformations and feature engineering in training and serving.
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        preprocessing_fn=configs.PREPROCESSING_FN
    )


@lru_cache(maxsize=None)
def trainer(
        example_gen=example_gen(),
        schema_gen=schema_gen(),
        # transform=transform()
):
    return Trainer(
        run_fn=configs.RUN_FN,
        examples=example_gen.outputs['examples'],
        # transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        # transform_graph=transform.outputs['transform_graph'],
#         train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
#         eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
        custom_executor_spec=executor_spec.ExecutorClassSpec(ai_platform_trainer_executor.GenericExecutor),
        custom_config={ai_platform_trainer_executor.TRAINING_ARGS_KEY: configs.GCP_AI_PLATFORM_TRAINING_ARGS}
    )


@lru_cache(maxsize=None)
def model_resolver():
    # Get the latest blessed model for model validation.
    return resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')


@lru_cache(maxsize=None)
def evaluator(
        example_gen=example_gen(),
        trainer=trainer(),
        model_resolver=model_resolver()
):
    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=features.LABEL_KEY)],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                # tfma.MetricConfig(
                #     class_name='BinaryAccuracy',
                #     threshold=tfma.MetricThreshold(
                #         value_threshold=tfma.GenericValueThreshold(
                #             lower_bound={'value': configs.EVAL_ACCURACY_THRESHOLD}),
                #         change_threshold=tfma.GenericChangeThreshold(direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                #                                                      absolute={'value': -1e-10})
                #     )
                # )
                tfma.MetricConfig(class_name='RootMeanSquaredError'),
                tfma.MetricConfig(class_name='MeanAbsolutePercentageError')
            ])
        ])

    return Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=eval_config
    )


@lru_cache(maxsize=None)
def pusher(
        trainer=trainer()
):
    return Pusher(
        model=trainer.outputs['model'],
        # model_blessing=evaluator.outputs['blessing'],
        # push_destination=pusher_pb2.PushDestination(filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=configs.SERVING_MODEL_DIR)),
        custom_executor_spec=executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor),
        custom_config={ai_platform_pusher_executor.SERVING_ARGS_KEY: configs.GCP_AI_PLATFORM_SERVING_ARGS}
    )
