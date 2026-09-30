"""
modules/feature_engineering/  — package version of the former
modules/feature_engineering.py (2,144 lines split into ~250-line
files, one per pipeline stage).

Backward compatibility: streamlit_app.py imports

    from modules.feature_engineering import intelligent_feature_engineering

That import keeps working unchanged — this __init__.py re-exports the
same function from pipeline.py. No other file needs to change.

If anything else in your project imports one of the individual stage
functions directly (e.g. intelligent_feature_selection), those are
re-exported below too.
"""

from .selection import intelligent_feature_selection
from .datetime_features import datetime_feature_engineering
from .categorical import intelligent_categorical_engineering
from .numeric import intelligent_numeric_engineering, smart_interaction_features
from .polynomial import intelligent_polynomial_features
from .ranking import intelligent_feature_ranking
from .multicollinearity import intelligent_multicollinearity_removal
from .scaling_pca import intelligent_scaling_pca
from .pipeline import intelligent_feature_engineering

__all__ = [
    "intelligent_feature_selection",
    "datetime_feature_engineering",
    "intelligent_categorical_engineering",
    "intelligent_numeric_engineering",
    "smart_interaction_features",
    "intelligent_polynomial_features",
    "intelligent_feature_ranking",
    "intelligent_multicollinearity_removal",
    "intelligent_scaling_pca",
    "intelligent_feature_engineering",
]
