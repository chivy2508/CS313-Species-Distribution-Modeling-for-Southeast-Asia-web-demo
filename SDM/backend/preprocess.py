import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if 'aspect' in df.columns:
        df['aspect_sin'] = np.sin(np.radians(df['aspect']))
        df['aspect_cos'] = np.cos(np.radians(df['aspect']))
        df = df.drop(columns=['aspect'])

    return df
