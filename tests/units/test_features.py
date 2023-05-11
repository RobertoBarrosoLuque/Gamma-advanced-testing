import pytest
from src.train.features.base import BaseFeature
from src.train.features import FEATURES_LIST
from src.train.features.features import Out
from tests.units.dummy_data import OUT_DATA
import numpy as np

OUT_FEATURE = Out()


class TestFeaturesUsage:
    @pytest.mark.parametrize("method", ["name", "fit", "fit_transform", "get_feature_names", "transform"])
    @pytest.mark.parametrize("feature", FEATURES_LIST)
    def test_feature_methods(self, feature, method):
        assert hasattr(feature, method)
        assert callable(getattr(feature, method))

    @pytest.mark.parametrize("method", ["name", "fit", "fit_transform", "get_feature_names"])
    def test_base_feature(self, method):
        assert hasattr(BaseFeature, method)
        assert callable(getattr(BaseFeature, method))


@pytest.mark.parametrize("x", OUT_DATA)
def test_out_feature(x):
    res = OUT_FEATURE.transform(X=x)
    isinstance(res, np.ndarray)





