import pytest
from src.train.features.base import BaseFeature
from src.train.features import FEATURES_LIST


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






