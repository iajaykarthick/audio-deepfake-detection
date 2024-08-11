import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway

def perform_statistical_tests(df, target_column='real_or_fake'):
    """
    Perform statistical tests to compare the distribution of features between real and fake audio samples.
    """
    real_data = df[df[target_column] == 'R']
    fake_data = df[df[target_column] != 'R']

    results = []

    for column in df.columns:
        try:
            if column not in ['audio_id', target_column]:
                real_values = real_data[column].dropna()
                fake_values = fake_data[column].dropna()
                
                if len(real_values) == 0 or len(fake_values) == 0:
                    continue

                if np.all(real_values == real_values.iloc[0]) or np.all(fake_values == fake_values.iloc[0]):
                    continue

                t_stat, t_p_val = ttest_ind(real_values, fake_values, equal_var=False)
                u_stat, u_p_val = mannwhitneyu(real_values, fake_values)
                effect_size = (real_values.mean() - fake_values.mean()) / np.sqrt((real_values.var() + fake_values.var()) / 2)
                f_stat, f_p_val = f_oneway(real_values, fake_values)

                results.append({
                    'feature': column,
                    't_stat': t_stat,
                    't_p_val': t_p_val,
                    'u_stat': u_stat,
                    'u_p_val': u_p_val,
                    'effect_size': effect_size,
                    'f_stat': f_stat,
                    'f_p_val': f_p_val
                })

        except Exception as e:
            print(f'Error in performing statistical tests for {column}: {e}')
        
    results_df = pd.DataFrame(results)
    return results_df
