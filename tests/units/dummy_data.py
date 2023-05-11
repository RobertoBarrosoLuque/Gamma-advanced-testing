import pandas as pd

OUT_DATA = [
    pd.DataFrame([{'outside.sideline': False, 'outside.baseline': True}, {'outside.sideline': False, 'outside.baseline': True}]),
    pd.DataFrame(
        [{'outside.sideline': True, 'outside.baseline': True}, {'outside.sideline': False, 'outside.baseline': True}])
]