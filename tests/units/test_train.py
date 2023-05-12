import numpy as np
import pytest
from src.train.data import load_raw_data
from src.train.train import (train, cross_validate, evaluate, hyperopt, log_experiment,
                             train_test_split_, get_best_estimator_and_cv_log_loss)
from src.train.model import get_model, get_param_grid
from src.utils import SklearnEstimator


@pytest.fixture()
def x_y_data():
    return train_test_split_(load_raw_data())


@pytest.fixture()
def estimator():
    return get_model()


@pytest.fixture()
def data():
    return load_raw_data()


def _assert_estimator_score(estimator, score):
    # Assert
    assert isinstance(estimator, SklearnEstimator)
    assert isinstance(score, np.ndarray)
    assert hasattr(estimator, 'fit')
    assert hasattr(estimator, 'predict')


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize('optimize', [True, False])
def test_train(estimator, data, optimize, log):
    #  Act
    test_log_loss, cv_log_losses = train(
        estimator=estimator, data=data, optimize=optimize,
        log=log
    )

    #  Assert
    assert isinstance(test_log_loss, float)
    assert isinstance(cv_log_losses, np.ndarray)


def test_run_cross_validation(estimator, x_y_data):
    X_train, _, y_train, _ = x_y_data
    estimator, cv_score = cross_validate(estimator=estimator, X=X_train,
                                         y=y_train)
    _assert_estimator_score(estimator, cv_score)


def test_hyperopt(estimator, x_y_data):
    # Arrange
    param_grid = get_param_grid()
    X_train, _, y_train, _ = x_y_data

    # Act
    estimator, score = hyperopt(X=X_train, y=y_train, param_grid=param_grid, estimator=estimator)

    # Assert
    _assert_estimator_score(estimator, score)


@pytest.mark.parametrize('optimize', [True, False])
def test_evaluate(x_y_data, optimize, estimator):
    X_train, _, y_train, _ = x_y_data
    estimator, _ = get_best_estimator_and_cv_log_loss(X_train=X_train, y_train=y_train,
                                                      optimize=optimize, estimator=estimator)

    X_train, _, y_train, _ = x_y_data
    log_val = evaluate(X=X_train, y=y_train, estimator=estimator)
    assert isinstance(log_val, float)

