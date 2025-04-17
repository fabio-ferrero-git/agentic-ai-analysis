"""
Statistical Analysis Utilities

This module contains functions for statistical analysis of user interaction data,
questionnaire responses, and system comparisons.

It provides tools for:
    - Analyzing user actions, message replies and user activity
    - Processing questionnaire data (durations and responses)
    - Performing statistical tests (Kruskal-Wallis, Mann-Whitney U, TOST)
    - Conducting regression analysis and correlation comparisons
    - Visualizing results through appropriate plots

All p-values in test results are appropriately formatted with significance indicators
and corrected for multiple comparisons when applicable.
"""

import itertools
from statistics import mean
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import analysis_functions.utils as u


def n_action_in_interface(n_actions: dict[int, list[int]],
                          stddev_actions: dict[int, np.float64], plot=False):
    """
    Calculates the average number of actions performed in 5 minutes of interface interaction for each system.

    Parameters:
        n_actions (dict): A dict with system identifiers as keys and lists of action counts as values.
        stddev_actions (dict): A dict with system identifiers as keys and standard deviations of actions as values.
        plot (bool, optional): When True, generates a bar plot of the average number of actions. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the average number of actions and their stddev for each system.
    """
    n_actions_in_5min = {f'System_{k}': [np.mean(v)] for k, v in n_actions.items()}
    if plot:
        plt.bar([0, 1, 2], [x[0] for x in n_actions_in_5min.values()])
        plt.xticks([0, 1, 2], ['System_0', 'System_1', 'System_2'])
        plt.show()
    # Also add standard deviations (already computed in dict stddev_actions)
    for k, v in stddev_actions.items():
        n_actions_in_5min[f'System_{k}'].append(v)
    return pd.DataFrame.from_dict(n_actions_in_5min, orient='index', columns=['mean', 'stddev'])



def avg_message_replied(replies:dict[int, list[float]], plot=False, color='blue'):
    """
    Calculates the average number of answered messages (normalized to [0,1]) for each system.

    Parameters:
        replies (dict): A dictionary with system identifiers as keys and lists of reply rates as values.
        plot (bool, optional): When True, generates a bar plot of the average reply rates. Defaults to False.
        color (str, optional): The color to use for the bars in the plot. Defaults to 'blue'.

    Returns:
        pd.DataFrame: A DataFrame containing the average message reply rates and their stddev for each system.
    """
    if plot:
        plt.bar([0, 1, 2], [np.mean(v) for _, v in replies.items()], color=color)
        plt.xticks([0, 1, 2], ['System_0', 'System_1', 'System_2'])
        plt.show()
    return pd.DataFrame.from_dict({
        sys : [f'{np.mean(np.array(v)):.4} ({np.std(np.array(v)):.4})'] for sys, v in replies.items()
    }, orient='index', columns=['mean (stddev)'])



def hand_vs_llm_actions(replies:dict[int, list[float]]):
    """
    Visualizes the distribution of user actions versus LLM actions for each system.

    Parameters:
        replies (dict): A dictionary containing pairs of user and LLM action values for each system.

    Returns:
        None: This function generates a plot but does not return any values.
    """
    labels = ["USER", "LLM"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, (user_val, llm_val) in enumerate(replies):
        bars = axes[i].bar(labels, [user_val, llm_val], color=['blue', 'orange'])
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[i].set_title(f'Distribution System {i}\n')
        axes[i].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()



def questionnaires_duration(questionnaires_durations: dict[int, dict[str, list[float]]]):
    """
    Parameters:
        questionnaires_durations (dict): A nested dictionary with system identifiers as outer keys,
                                        questionnaire name identifiers as inner keys, and lists of
                                        completion durations (in seconds) as values.

    Returns:
        pd.DataFrame: A DataFrame containing the average questionnaire duration for each system,
                     formatted as both seconds and minutes.
    """
    dur_df = pd.DataFrame.from_dict({
        q_phase: [
            (lambda duration_val: f'{duration_val:.3f}s ({duration_val/60:.3f}m)' if duration_val else '--')
            (mean(questionnaires_durations.get(cond, {}).get(q_phase, [0])))
            for cond in questionnaires_durations.keys()
        ] + [
            (lambda total_mean: f'{total_mean:.3f}s ({total_mean/60:.3f}m)' if total_mean else '--')(
                mean([
                    val for cond in questionnaires_durations.keys()
                    for val in questionnaires_durations.get(cond, {}).get(q_phase, [0])
                ])
            )
        ]
        for q_phase in u.quest_phases
    }, orient='index')
    dur_df.columns = ([f'sys_{n}' for n in list(questionnaires_durations.keys())] +
                      ['Total Mean (without considering systems)'])
    return dur_df



def format_p_values(values, n_hypotheses=18):
    """
    Formats p-values with significance indicators after applying Bonferroni-Holm correction.

    Parameters:
        values (float, list, tuple, np.ndarray): The p-value(s) to format.
        n_hypotheses (int, optional): Total number of hypotheses being tested. Defaults to 18.

    Returns:
        str or list: Formatted p-value(s) with significance indicators:
                    (***) for p < 0.001
                    (**) for p < 0.005
                    (*) for p < 0.05
    """
    def apply_bonferroni_holm(p_values, n_hyp):
        """
        Apply Bonferroni-Holm correction to a set of p-values.
        Args:
            p_values (float, list, tuple, np.ndarray): The p-values to apply
            n_hyp: Total number of hypotheses being tested
        """
        p_array = np.array(p_values, dtype=float)

        # Sort p-values
        sorted_indices = np.argsort(p_array)

        # Apply Bonferroni-Holm
        adj_p = np.ones_like(p_array)
        for i, idx in enumerate(sorted_indices):
            adjustment = n_hyp - i
            adj_p[idx] = min(p_array[idx] * adjustment, 1.0)

        return adj_p

    def format_single_p(adj_p):
        """
        Format a single p-value with significance indicators.
        """
        return f"{adj_p:.9f} " + \
            ("(***)" if adj_p < 0.001 else
             "(**)" if adj_p < 0.005 else
             "(*)" if adj_p < 0.05 else "")

    if isinstance(values, (list, tuple, np.ndarray)):
        # Handle multiple p-values
        adjusted_p_values = apply_bonferroni_holm(values, n_hypotheses)

        # Format each p-value
        formatted = [format_single_p(adj_p) for p, adj_p in zip(values, adjusted_p_values)]

        # Return in the same format as input
        if isinstance(values, list):
            return formatted
        elif isinstance(values, tuple):
            return tuple(formatted)
        else:
            return np.array(formatted)
    else:
        # Single p-value, apply standard Bonferroni
        adjusted_p = min(values * n_hypotheses, 1.0)
        return format_single_p(adjusted_p)



def equivalence_testing(questionnaires_results : dict[int, dict[str, list[list[int]]]],
                        study_questions : dict[str: list[str]]):
    """
    Performs non-parametric equivalence testing on questionnaire results for each system combination.
    Applies Bonferroni correction to p-values using format_p_values function.

    Parameters:
        questionnaires_results (dict):  A nested dictionary containing questionnaire results with system identifiers
                                        as outer keys, questionnaire name identifiers as inner keys, and
                                        lists of questionnaire results as values.
        study_questions (dict): A dictionary mapping questionnaire names to lists of questions.

    Returns:
        dict[str, pd.DataFrame]: A dictionary of DataFrames containing non-parametric equivalence
                               testing results for each questionnaire, organized by system combinations
                               (System0-System1, System0-System2, System1-System2).
    """
    tost_dfs = {}
    combinations = list(itertools.combinations(questionnaires_results.keys(), 2))
    for q_name in u.quest_names:
        data = {'Questions': study_questions[q_name]}
        for c in combinations:
            p_values = []
            means_stds = []
            bound = 10 if q_name == 'tlx_questionnaire' else 0.5
            if questionnaires_results[c[0]].get(q_name) and questionnaires_results[c[1]].get(q_name):
                for col in range(np.array(questionnaires_results[c[0]][q_name]).shape[1]):

                    col0 = np.array(questionnaires_results[c[0]][q_name])[:, col]
                    col1 = np.array(questionnaires_results[c[1]][q_name])[:, col]
                    m0, m1 = np.mean(col0), np.mean(col1)
                    std0, std1 = np.std(col0), np.std(col1)
                    p_values.append(non_parametric_tost(col0, col1, bound))
                    means_stds.append(f'{m0:.2f} ({std0:.2f}), {m1:.2f} ({std1:.2f})')

                data[f'Sys{c[0]}-Sys{c[1]} mean (std)'] = means_stds
                data[f'Sys{c[0]}-Sys{c[1]} (bound = {bound})'] = format_p_values(p_values)
        tost_dfs[q_name] = pd.DataFrame(data)
    return tost_dfs



def linear_regression(x : List[Union[int, float]],
                      y : List[Union[int, float]],
                      plot: bool = False):
    """
    Computes simple linear regression between two variables.

    Parameters:
        x (List[Union[int, float]]): The independent variable values.
        y (List[Union[int, float]]): The dependent variable values.
        plot (bool, optional): When True, generates a plot of the regression. Defaults to False.

    Returns:
        tuple: Contains:
            - R-squared value of the regression model (float, rounded to 5 decimal places)
            - Correlation coefficient (float, rounded to 5 decimal places)
    """
    X = np.array(x).flatten()
    Y = np.array(y).flatten()

    data = {
        'X': X,  # Ensure X and Y are properly defined before this
        'Y': Y
    }
    df_model = pd.DataFrame(data)
    X = df_model['X'].values.reshape(-1, 1)
    Y = df_model['Y'].values

    model = LinearRegression()
    model.fit(X, Y)

    Y_pred = model.predict(X)

    r_2 = r2_score(Y, Y_pred)
    intercept = model.intercept_
    slope = model.coef_
    correlation_c = float((np.sign(slope) * np.sqrt(r_2))[0])
    residuals = Y - Y_pred

    return round(r_2, ndigits=5), round(correlation_c, ndigits=5),



def full_regression_analysis(pre_task_aggregation_pairs : list[tuple],
                             pre_task_aggregation : dict[int, dict[str, list[float]]],
                             tia_post_task_aggregation : dict[int, dict[str, list[float]]],
                             tlx_aggregation : dict[int, list[list[int]]],
                             grid_questionnaire_aggregation : dict[int, list[list[int]]],
                             tlx_aggregation_unique : dict[int, ndarray]):
    """
    Performs comprehensive linear regression analysis between pre-task and post-task questionnaire measures.

    Parameters:
        pre_task_aggregation_pairs (list[tuple]): System-PreTaskQuestionnaire pairs (e.g., 0-desirability_of_control).

        pre_task_aggregation (dict): A nested dictionary containing Pre-task questionnaire aggregation results
                                     with system identifiers as outer keys, questionnaire identifiers as inner
                                     keys, and questionnaire aggregation results as values.
                                     e.g., {0 : {'tia_pre_task' : [v1,..., vn]}} where 0 means 'System0'

        tia_post_task_aggregation (dict): A nested dictionary containing TiA-Post-Task aggregation results
                                          with system identifiers as outer keys, questionnaire aggregation
                                          identifiers as inner keys, and questionnaire aggregation results
                                          as values.
                                          e.g., {0 : {'familiarity' : [v1, ..., vn]}} where 0 means 'System0'

        tlx_aggregation (dict): NASA-TLX aggregations results with system identifiers as outer keys and
                                questionnaire aggregation results as values.
                                NOTE: Each Nasa-TLX question is an aggregation.
                                e.g., {0 : [[values_1], ... , [values_n]]} where 0 means 'System0'

        grid_questionnaire_aggregation (dict): Grid-Questionnaire aggregations results with system identifiers
                                               as outer keys and questionnaire aggregation results as values.
                                               NOTE: Each Grid-Questionnaire question is an aggregation.
                                              e.g., {0 : [[values_1],...,[values_n]]} where 1 means 'System1'


        tlx_aggregation_unique (dict):  NASA-TLX full aggregation results with system identifiers as outer keys
                                        and questionnaire aggregation results as values.
                                        NOTE: In this case, all NASA-TLX questions form an aggregation.
                                        e.g., {0 : [v1, ..., vn]} where 0 means 'System 0'
    Returns:
        pd.DataFrame: A DataFrame containing correlation coefficients for each PRE-POST task questionnaire pair,
                      organized by system.
    """
    sys_str = 'System_'
    all_results = {}

    for sys, pre in pre_task_aggregation_pairs:
        sys_key = f'{sys_str}{sys}'
        all_results.setdefault(sys_key, {})

        # TIA post-task
        for post in tia_post_task_aggregation[sys].keys():
            X = pre_task_aggregation[sys][pre]
            Y = tia_post_task_aggregation[sys][post]
            row_label = f'PRE: {pre}  --  POST: tia {post}'
            _, correlation_c = linear_regression(X, Y)
            all_results[sys_key][row_label] = correlation_c

        # TLX aggregation
        for post in range(len(tlx_aggregation[sys])):
            X = pre_task_aggregation[sys][pre]
            Y = tlx_aggregation[sys][post]
            row_label = f'PRE: {pre}  --  POST: nasa-tlx question n.{post}'
            _, correlation_c = linear_regression(X, Y)
            all_results[sys_key][row_label] = correlation_c


        X_tlx_unique = pre_task_aggregation[sys][pre]
        Y_tlx_unique = tlx_aggregation_unique[sys]
        row_label_tlx_unique = f'PRE: {pre}  --  POST: nasa-tlx full'
        _, correlation_c = linear_regression(X_tlx_unique, Y_tlx_unique)
        all_results[sys_key][row_label_tlx_unique] = correlation_c


        # GRID questionnaire
        if grid_questionnaire_aggregation.get(sys) is not None:
            for post in range(len(grid_questionnaire_aggregation[sys])):
                X = pre_task_aggregation[sys][pre]
                Y = grid_questionnaire_aggregation[sys][post]
                row_label = f'PRE: {pre}  --  POST: grid question n.{post}'
                _, correlation_c = linear_regression(X, Y)
                all_results[sys_key][row_label] = correlation_c


    # Create DataFrame
    all_row_labels = {label for sys_results in all_results.values() for label in sys_results}
    df = pd.DataFrame(index=sorted(all_row_labels), columns=sorted(all_results.keys()))

    # Fill DataFrame
    for sys_key, sys_results in all_results.items():
        for row_label, result in sys_results.items():
            df.at[row_label, sys_key] = result

    return df.infer_objects(copy=False).fillna('--')



def compare_correlations(r1 : Union[str, float], r2 : Union[str, float],
                         n1 : int, n2, alpha=0.05):
    """
    Compares two independent correlation coefficients using Fisher's Z transformation.

    Parameters:
        r1 (Union[str, float]): First correlation coefficient.
        r2 (Union[str, float]): Second correlation coefficient.
        n1 (int): Sample size for first correlation.
        n2 (int): Sample size for second correlation.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        str: Formatted p-value resulting from the comparison using Fisher's Z transformation.
    """
    if r1 == '--' or r2 == '--':
        return '--'
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Standard error
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

    # Z-score
    z_score = (z1 - z2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Check significance
    significant = p_value < alpha

    return format_p_values(p_value)



def compare_systems_correlations(df_reg : pd.DataFrame, n : dict[int, int]):
    """
    Compare correlation coefficients using Fisher's Z transformation, for each system combinations
    System 0 - System 1
    System 0 - System 2
    System 1 - System 2

    Args:
        df_reg: pd.DataFrame with correlation coefficients
        n: user study sample

    Returns:
        pd.DataFrame with all p-values after performing comparison using Fisher's Z transformation'
    """
    combinations = list(itertools.combinations([0, 1, 2], 2))

    results = []

    for idx, row in df_reg.iterrows():
        name = row.name
        row_data = {'name': name}

        for c1, c2 in combinations:
            column_name = f'System{c1} - System{c2}'
            row_data[column_name] = compare_correlations(r1=row[f'System_{c1}'], r2=row[f'System_{c2}'],
                                                         n1=n[c1], n2=n[c2])

        results.append(row_data)

    # Create DataFrame from the results
    df_results = pd.DataFrame(results)

    # Set 'name' as the index if you want
    df_results.set_index('name', inplace=True)
    return df_results



def kruskal_wallis(questionnaires_results : dict[int, dict[str, list[list[int]]]],
                   study_questions : dict[str: list[str]]):
    """
    Performs the Kruskal-Wallis H-test, a non-parametric alternative to ANOVA, on questionnaire results.
    Applies Bonferroni correction to p-values using format_p_values function.

    Parameters:
        questionnaires_results (dict): A nested dictionary containing questionnaire results for each system.
        study_questions (dict): A dictionary mapping questionnaire names to lists of questions.

    Returns:
        dict[str, pd.DataFrame]: A dictionary of DataFrames containing Kruskal-Wallis H-test results
                               for each questionnaire, including means, standard deviations, and
                               Bonferroni-corrected p-values.
    """
    from scipy.stats import kruskal

    kw_dfs = {}
    for q_name in u.quest_names:
        # Get groups that have this questionnaire
        groups = [q[q_name] for q in questionnaires_results.values() if q_name in q]
        if len(groups) < 2:
            continue  # Need at least 2 groups

        p_values = []

        # Test each question separately
        for i in range(len(study_questions[q_name])):
            # Extract data for this question from each group
            columns = []
            for group in groups:
                group_array = np.array(group)
                columns.append(group_array[:, i])

            # Perform Kruskal-Wallis test
            try:
                _, p = kruskal(*columns)
                p_values.append(p)
            except ValueError:  # For identical distributions
                p_values.append(1.0)

        kw_dfs[q_name] = pd.DataFrame({
            'Questions': study_questions[q_name],
            **{f'mean (std) SYS_{sys}': [f"{m:.2f} ({std:.2f})" for m, std in zip(
                np.mean(questionnaires_results[sys][q_name], axis=0),
                np.std(questionnaires_results[sys][q_name], axis=0)
            )]
               for sys in list(questionnaires_results.keys()) if questionnaires_results[sys].get(q_name)},
            'p_values': format_p_values(p_values)
        })

    return kw_dfs



def mann_whitney(questionnaires_results, study_questions):
    """
    Performs the Mann-Whitney U test, a non-parametric alternative to t-test, for pairwise system comparisons.
    Applies Bonferroni correction to p-values using format_p_values function.

    Parameters:
        questionnaires_results (dict): A nested dictionary containing questionnaire results for each system.
        study_questions (dict): A dictionary mapping questionnaire names to lists of questions.

    Returns:
        dict[str, pd.DataFrame]: A dictionary of DataFrames containing Mann-Whitney U test results
                               for each questionnaire, organized by system pairs, with
                               Bonferroni-corrected p-values.
    """
    from scipy.stats import mannwhitneyu

    mw_dfs = {}
    combinations = list(itertools.combinations(questionnaires_results.keys(), 2))

    for q_name in u.quest_names:
        data = {'Questions': study_questions[q_name]}
        for c in combinations:
            if questionnaires_results[c[0]].get(q_name) and questionnaires_results[c[1]].get(q_name):
                # Convert to numpy arrays for consistent indexing
                group0 = np.array(questionnaires_results[c[0]][q_name])
                group1 = np.array(questionnaires_results[c[1]][q_name])

                p_values = []

                # Test each question separately
                for i in range(group0.shape[1]):
                    col0 = group0[:, i]
                    col1 = group1[:, i]

                    # Perform Mann-Whitney U test
                    try:
                        _, p = mannwhitneyu(col0, col1, alternative='two-sided')
                        p_values.append(p)
                    except ValueError:  # For identical distributions
                        p_values.append(1.0)

                # Format with same structure as original
                p_values_formatted = format_p_values(p_values)
                m0, m1 = np.mean(group0, axis=0), np.mean(group1, axis=0)
                std0, std1 = np.std(group0, axis=0), np.std(group1, axis=0)

                data[f'Sys{c[0]}-Sys{c[1]} mean (std)'] = [
                    f'{m0[i]:.2f} ({std0[i]:.2f}), {m1[i]:.2f} ({std1[i]:.2f})'
                    for i in range(len(p_values))
                ]
                data[f'Sys{c[0]}-Sys{c[1]} p_val'] = [f'p={x}' for x in p_values_formatted]

        mw_dfs[q_name] = pd.DataFrame(data)

    return mw_dfs



def np_tost_test(x, y, eps, alpha=0.05):
    """
    Implements a non-parametric Two One-Sided Tests (TOST) procedure for equivalence testing.

    Parameters:
        x (array-like): First data series.
        y (array-like): Second data series.
        eps (float): Equivalence bound (epsilon).
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        float: p-value resulting from the non-parametric TOST procedure.
    """
    from scipy.stats import mannwhitneyu

    # Convert to numpy arrays
    x_array = np.array(x)
    y_array = np.array(y)

    # Test 1: Is effect at least -eps? (x > y - eps)
    y_minus_eps = y_array - eps
    try:
        _, p_lower = mannwhitneyu(x_array, y_minus_eps, alternative='greater')
    except ValueError:
        p_lower = 1.0 if np.median(x_array) <= np.median(y_minus_eps) else 0.0

    # Test 2: Is effect at most eps? (y > x - eps)
    x_minus_eps = x_array - eps
    try:
        _, p_upper = mannwhitneyu(y_array, x_minus_eps, alternative='greater')
    except ValueError:
        p_upper = 1.0 if np.median(y_array) <= np.median(x_minus_eps) else 0.0

    # Overall p-value is maximum of the two
    p_value = max(p_lower, p_upper)

    return p_value



def non_parametric_tost(x, y, bound):
    """
    Wrapper
    """
    # Simply call np_tost_test with the provided bound
    return np_tost_test(x, y, eps=bound, alpha=0.05)



def np_test_toast_post_task_aggregation(tia_post_task_aggregation):
    """
    Performs non-parametric t-tests and TOST tests on post-task aggregation data across systems.

    Parameters:
        tia_post_task_aggregation (dict): A nested dictionary containing TiA post-task aggregations by system.

    Returns:
        tuple: Two DataFrames:
            - DataFrame with Mann-Whitney U test results (non-parametric t-test alternative)
            - DataFrame with non-parametric TOST test results
    """
    from scipy.stats import mannwhitneyu

    res_dict_mw = {}  # Mann-Whitney (instead of t-test)
    res_dict_tost = {}  # Non-parametric TOST
    combinations = list(itertools.combinations(tia_post_task_aggregation.keys(), 2))

    for sys_a, sys_b in combinations:
        sys_couple = f'System {sys_a} - System {sys_b}'
        res_dict_mw[sys_couple] = {}
        res_dict_tost[sys_couple] = {}

        for agg in tia_post_task_aggregation[0].keys():
            mean_a = np.mean(tia_post_task_aggregation[sys_a][agg])
            mean_b = np.mean(tia_post_task_aggregation[sys_b][agg])
            std_a = np.std(tia_post_task_aggregation[sys_a][agg])
            std_b = np.std(tia_post_task_aggregation[sys_b][agg])

            # Mann-Whitney U test
            try:
                _, p_value = mannwhitneyu(
                    tia_post_task_aggregation[sys_a][agg],
                    tia_post_task_aggregation[sys_b][agg],
                    alternative='two-sided'
                )
            except ValueError:  # For identical distributions
                p_value = 1.0

            formatted_p = format_p_values(p_value)
            res_dict_mw[sys_couple][agg] = f"{mean_a:.2f} ({std_a:.2f}), {mean_b:.2f} ({std_b:.2f}) - p={formatted_p}"

            # Non-parametric TOST
            tost_p = non_parametric_tost(
                tia_post_task_aggregation[sys_a][agg],
                tia_post_task_aggregation[sys_b][agg],
                bound=0.5
            )
            formatted_tost_p = format_p_values([tost_p])[0]
            res_dict_tost[sys_couple][
                agg] = f"{mean_a:.2f} ({std_a:.2f}), {mean_b:.2f} ({std_b:.2f}) - p={formatted_tost_p}"

    results_df_mw = pd.DataFrame.from_dict(res_dict_mw, orient='index')
    results_df_tost = pd.DataFrame.from_dict(res_dict_tost, orient='index')

    return results_df_mw, results_df_tost



def analyze_tia_post_task_aggregation(tia_post_task_aggregation):
    """
    Analyzes the aggregated TiA-Post Task scores across systems using the Kruskal-Wallis test.

    Parameters:
        tia_post_task_aggregation (dict): A nested dictionary with system identifiers as keys
                                         and aggregated TiA Post Task scores as values.

    Returns:
        pd.DataFrame: A DataFrame containing the Kruskal-Wallis test results, including means,
                     standard deviations, and Bonferroni-corrected formatted p-values for each aggregation.
    """
    from scipy.stats import kruskal

    df_data = []
    for agg in tia_post_task_aggregation[0].keys():
        data = [tia_post_task_aggregation[sys][agg] for sys in range(3)]
        _, p_val = kruskal(*data)
        means = [f'{np.mean(data[sys]):.2f} ({np.std(data[sys]):.2f})' for sys in range(3)]
        df_data.append({
            'aggregation': agg,
            **{f'mean_sys_{sys}': means[sys] for sys in range(3)},
            'p_value': format_p_values(p_val)
        })
    return pd.DataFrame(df_data)



def analyze_tlx_aggregate(tlx_aggregation_unique):
    """
    Analyzes the aggregated NASA-TLX scores across systems using non-parametric statistical tests.
    Applies Bonferroni correction to p-values using format_p_values function for Kruskal-Wallis
    and Mann-Whitney U test results.

    Parameters:
        tlx_aggregation_unique (dict): A dictionary with system identifiers as keys
                                      and aggregated NASA-TLX scores as values.

    Returns:
        tuple: Three DataFrames:
            - DataFrame with Kruskal-Wallis test results (non-parametric ANOVA) with Bonferroni-corrected p-values
            - DataFrame with Mann-Whitney U test results (pairwise comparisons) with Bonferroni-corrected p-values
            - DataFrame with non-parametric TOST test results (equivalence testing) with Bonferroni-corrected p-values
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import kruskal, mannwhitneyu

    # Calculate mean and std for each system
    system_stats = {
        f'System_{sys}': {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        for sys, scores in tlx_aggregation_unique.items()
    }

    # Perform Kruskal-Wallis test (non-parametric ANOVA)
    groups = [tlx_aggregation_unique[sys] for sys in sorted(tlx_aggregation_unique.keys())]
    try:
        _, p_value = kruskal(*groups)
    except ValueError:
        p_value = 1.0

    # Create ANOVA-like DataFrame
    anova_df = pd.DataFrame({
        'Questions': ['NASA-TLX Aggregate'],
        **{f'mean (std) SYS_{sys}': [
            f"{system_stats[f'System_{sys}']['mean']:.2f} ({system_stats[f'System_{sys}']['std']:.2f})"]
           for sys in sorted(tlx_aggregation_unique.keys())},
        'p_values': [format_p_values(p_value)]
    })

    # Perform Mann-Whitney tests (non-parametric t-tests)
    combinations = list(itertools.combinations(sorted(tlx_aggregation_unique.keys()), 2))
    mw_data = {'Questions': ['NASA-TLX Aggregate']}

    for c in combinations:
        mean_a = system_stats[f'System_{c[0]}']['mean']
        std_a = system_stats[f'System_{c[0]}']['std']
        mean_b = system_stats[f'System_{c[1]}']['mean']
        std_b = system_stats[f'System_{c[1]}']['std']

        try:
            _, p_value = mannwhitneyu(
                tlx_aggregation_unique[c[0]],
                tlx_aggregation_unique[c[1]],
                alternative='two-sided'
            )
        except ValueError:
            p_value = 1.0

        formatted_p = format_p_values(p_value)

        mw_data[f'Sys{c[0]}-Sys{c[1]} mean (std)'] = [f'{mean_a:.2f} ({std_a:.2f}), {mean_b:.2f} ({std_b:.2f})']
        mw_data[f'Sys{c[0]}-Sys{c[1]} p_val'] = [f'p={formatted_p}']

    mw_df = pd.DataFrame(mw_data)

    # Perform TOST tests (non-parametric equivalence testing)
    tost_data = {'Questions': ['NASA-TLX Aggregate']}
    bound = 10  # Same as for individual TLX questions

    for c in combinations:
        mean_a = system_stats[f'System_{c[0]}']['mean']
        std_a = system_stats[f'System_{c[0]}']['std']
        mean_b = system_stats[f'System_{c[1]}']['mean']
        std_b = system_stats[f'System_{c[1]}']['std']

        # Non-parametric TOST
        p_value = non_parametric_tost(
            tlx_aggregation_unique[c[0]],
            tlx_aggregation_unique[c[1]],
            bound=bound
        )

        formatted_p = format_p_values([p_value])[0]

        tost_data[f'Sys{c[0]}-Sys{c[1]} mean (std)'] = [f'{mean_a:.2f} ({std_a:.2f}), {mean_b:.2f} ({std_b:.2f})']
        tost_data[f'Sys{c[0]}-Sys{c[1]} (bound = {bound})'] = [formatted_p]

    tost_df = pd.DataFrame(tost_data)

    return anova_df, mw_df, tost_df



# Define key categories for evaluation metrics
correct_keys = ['correct_calendar_events', 'correct_todo_items']
wrong_keys = ['wrong_calendar_events', 'wrong_todo_items']
missing_keys = ['missing_calendar_events', 'missing_todo_items']


def precision(actions):
    """
    Calculate precision: correct / (correct + wrong)

    Args:
        actions: Dictionary containing action counts

    Returns:
        float: Precision score between 0 and 1
    """
    correct = sum(actions.get(k, 0) for k in correct_keys)
    wrong = sum(actions.get(k, 0) for k in wrong_keys)
    return correct / (correct + wrong) if (correct + wrong) else 0


def recall(actions):
    """
    Calculate recall: correct / (correct + missing + wrong)

    Args:
        actions: Dictionary containing action counts

    Returns:
        float: Recall score between 0 and 1
    """
    correct = sum(actions.get(k, 0) for k in correct_keys)
    missing = sum(actions.get(k, 0) for k in missing_keys)
    wrong = sum(actions.get(k, 0) for k in wrong_keys)
    tot = correct + missing + wrong
    return correct / tot if tot else 0