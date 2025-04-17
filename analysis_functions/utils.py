"""
Utility functions for processing user timeline data and questionnaire responses.

This module provides functions for extracting and analyzing user interaction data
from experimental logs. Functions handle various aspects of data processing including:

    - Retrieving user conditions and scenarios
    - Calculating time differences between events
    - Processing questionnaire responses into normalized formats
    - Extracting and validating free-text answers

Each function is designed to work with standardized data structures from the
experimental logging system, particularly with Timeline and LogEvent objects from logs package.
"""

import pandas as pd
from statistics import mean
from analysis_functions.logs import LogEvent, Timeline, AnswersEvent

#
# User study phases
#
all_phases = [
    'introduction',
    'before-interface',
    'desirability-of-control',
    'tia-pre-task',
    'sys-instruction',
    'interface',
    'tia-post-task',
    'nasa-tlx',
    'free-text',
    'grid-questionnaire',
    'end',
]

#
# Names of user study questionnaires
#
quest_names = [
    'desirability_of_control',
    'tia_pre_task',
    'tia_post_task',
    'tlx_questionnaire',
    'grid_questionnaire'
]

#
# Phases of user study questionnaires
#
quest_phases = [
    'desirability-of-control',
    'tia-pre-task',
    'tia-post-task',
    'nasa-tlx',
    'grid-questionnaire',
    'free-text',
]

#
# Likert scale mapping (from str to int)
#
to_likert_scale = {
    'Strongly Disagree' : 1,
    'Disagree': 2,
    'Neither Agree nor Disagree' : 3,
    'Agree' : 4,
    'Strongly Agree' : 5
}

#
# Pre-task questionnaires
#
pre_task_names = [
    'desirability_of_control',
    'tia_pre_task'
]

#
# Scores mapping for each questionnaires
#
scores_map = {
    'desirability_of_control' : {
        1: {'invert': False},
        2: {'invert': False},
        3: {'invert': False},
        4: {'invert': False},
        5: {'invert': False},
        6: {'invert': False},
        7: {'invert': True},
        8: {'invert': False},
        9: {'invert': False},
        10: {'invert': True},
        11: {'invert': False},
        12: {'invert': False},
        13: {'invert': False},
        14: {'invert': False},
        15: {'invert': False},
        16: {'invert': True},
        17: {'invert': False},
        18: {'invert': False},
        19: {'invert': True},
        20: {'invert': True},
    },
    'tia_pre_task' : {
        1: {'invert': True},
        2: {'invert': False},
        3: {'invert': False},
    },
    'tia_post_task' : {
        1: {'invert': False, 'aggr' : 'reliability_competence'},
        2: {'invert': False, 'aggr' : 'reliability_competence'},
        3: {'invert': True, 'aggr' : 'reliability_competence'},
        4: {'invert': False, 'aggr' : 'reliability_competence'},
        5: {'invert': True, 'aggr' : 'reliability_competence'},
        6: {'invert': False, 'aggr' : 'reliability_competence'},
        7: {'invert': False, 'aggr' : 'understanding_predictability'},
        8: {'invert': True, 'aggr' : 'understanding_predictability'},
        9: {'invert': False, 'aggr' : 'understanding_predictability'},
        10: {'invert': True, 'aggr' : 'understanding_predictability'},
        11: {'invert': False, 'aggr' : 'familiarity'},
        12: {'invert': False, 'aggr' : 'familiarity'},
        13: {'invert': False, 'aggr' : 'intention_of_developers'},
        14: {'invert': False, 'aggr' : 'intention_of_developers'},
        15: {'invert': False, 'aggr' : 'trust_in_automation'},
        16: {'invert': False, 'aggr' : 'trust_in_automation'},
    }
}

#
# Useful log names
#
log_start = 'study_started'
log_end = 'study_ended'
log_phase = 'phase_change'



def get_user_condition(dataframe: pd.DataFrame, uid: float):
    """
    Retrieves the condition assigned to a specific user from the provided dataframe.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing user data and assigned conditions
        uid (float): The unique identifier of the user to query

    Returns:
        The assigned condition value for the specified user if found, None otherwise
    """
    row = dataframe.loc[dataframe['user_id'] == uid, 'assigned_condition']
    return row.iloc[0].item() if not row.empty else None



def get_user_scenario(dataframe: pd.DataFrame, uid: float):
    """
    Retrieves the scenario assigned to a specific user from the provided dataframe.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing user data and scenario assignments
        uid (float): The unique identifier of the user to query

    Returns:
        A tuple containing (job, assigned_scenarios) for the specified user, or None if user not found
    """
    user_row = dataframe.loc[dataframe['user_id'] == uid]

    if user_row.empty:
        return None

    job = user_row['job'].iloc[0]
    assigned_scenarios = int(user_row['assigned_scenario'].iloc[0])

    return job, assigned_scenarios



def timestamp_diff(e1: LogEvent, e2: LogEvent):
    """
    Calculates the time difference between two log events.

    Parameters:
        e1 (LogEvent): The first log event
        e2 (LogEvent): The second log event

    Returns:
        float: The time difference in seconds between the timestamps of e1 and e2
    """
    return (e1.timestamp - e2.timestamp).total_seconds()



def get_questionnaires_duration(tml: Timeline, phases_logs: list[LogEvent], log_name=log_phase):
    """
    Calculates the duration spent by a user on each questionnaire phase.

    Parameters:
        tml (Timeline): The user's timeline of events
        phases_logs (list[LogEvent]): List of phase change log events
        log_name (str, optional): The name of the log event to search for. Defaults to 'phase_change'

    Returns:
        dict: A dictionary mapping questionnaire phases to their durations in seconds
    """
    return {
        e.log_information['to']: timestamp_diff(tml.get_next_log_after_event(log_name, e), e)
        for e in phases_logs if e.log_information['to'] in quest_phases
    }



def get_questionnaires_results(answer_events: list[AnswersEvent]):
    """
    Extracts questionnaire responses from user answer events.

    Parameters:
        answer_events (list[AnswersEvent]): List of events containing user questionnaire answers

    Returns:
        dict: A dictionary mapping questionnaire names to their normalized response values,
              converting text responses to numeric values using the Likert scale mapping
    """
    return {
        ae.event_name : [to_likert_scale[str(a)] if str(a) in to_likert_scale.keys() else int(a['percentage'])
                         for _, a in ae.answers.items()]
        for ae in answer_events if ae.event_name not in ['freeTextAnswers']
    }



def get_free_text_answers(answer_events: list[AnswersEvent]):
    """
    Extracts free-text responses from answer events, applying minimal validation.

    Parameters:
        answer_events (list[AnswersEvent]): List containing free text response events

    Returns:
        tuple: A tuple containing two elements (first answer, second answer),
               where answers shorter than 5 characters are replaced with None
    """
    return (answer_events[0].answers[0] if len(answer_events[0].answers[0].strip()) > 5 else None,
            answer_events[0].answers[1] if len(answer_events[0].answers[1].strip()) > 5 else None)



def fix_pre_task_questionnaire(to_fix: dict[int, dict[str, list[list[int]]]], u_cond: int):
    """
    Reorganizes questionnaire data to correct the structure of combined questionnaires.

    This function separates responses that were initially combined for visibility purposes.
    It redistributes the first 9 questions from the 'tia_pre_task' questionnaire to the
    'desirability_of_control' questionnaire, leaving only the final 3 questions in the
    'tia_pre_task' dataset.

    Parameters:
        to_fix (dict[int, dict[str, list[list[int]]]]): Nested dictionary containing
                questionnaire response data organized by condition and questionnaire type
        u_cond (int): The user condition identifier for which to perform the correction

    Note: IN-PLACE update
    """
    tia_pre_task = to_fix[u_cond]['tia_pre_task'][-1]
    to_fix[u_cond]['desirability_of_control'][-1].extend(tia_pre_task[:-3])
    to_fix[u_cond]['tia_pre_task'][-1] = (tia_pre_task[-3:])



def reverse_score(score : int, max_score : int = 5):
    """
    Reverses a score value on a Likert scale.

    Parameters:
        score (int): The original score to be reversed
        max_score (int, optional): The maximum possible score on the scale. Defaults to 5

    Returns:
        int: The reversed score value, calculated as (max_score - score + 1)
    """
    return max_score - score + 1


def get_user_score(scores : list[int], score_name : str):
    """
    Calculates a user's average score for a specific questionnaire, applying score inversion where needed.

    Parameters:
        scores (list[int]): List of individual question scores for the user
        score_name (str): The name of the questionnaire or scoring metric

    Returns:
        float: The mean score across all questions, with appropriate score inversion
               applied according to the questionnaire's scoring rule
    """
    return mean(reverse_score(s) if scores_map[score_name][i+1]['invert'] else s for i, s in enumerate(scores))



def get_aggregation_score(scores : list[list[int]], score_name : str):
    """
    Calculates the aggregated scores for multiple users on a specific questionnaire.

    Parameters:
        scores (list[list[int]]): Nested list where each inner list contains
                                 scores for a single user across all questions
        score_name (str): The name of the questionnaire or scoring metric

    Returns:
        list[float]: A list containing the mean score for each user
    """
    return [get_user_score(user_score, score_name) for user_score in scores]



def tia_post_task_aggregation_score(scores : list[list[int]]):
    """
    Calculates aggregated scores for the post-task questionnaire, grouped by aggregation categories.

    This function processes scores from the 'tia_post_task' questionnaire, grouping questions
    by their predefined aggregation categories and calculating mean scores for each category.

    Parameters:
        scores (list[list[int]]): Nested list where each inner list contains
                                 scores for a single user across all questions

    Returns:
        dict: A dictionary mapping aggregation category names to lists of user scores,
              where each list contains the mean score for a user in that category
    """
    aggr = {a: [] for a in set([x['aggr'] for x in scores_map['tia_post_task'].values()])}

    for aggr_name in aggr:
        filtered_scores = [
            [user_scores[q_num - 1] for q_num, meta in scores_map['tia_post_task'].items()
             if meta.get('aggr') == aggr_name]
            for user_scores in scores
        ]
        aggr[aggr_name] = [get_user_score(user_scores, 'tia_post_task') for user_scores in filtered_scores]

    return aggr



#
#
# Demographic
#
#
def get_demographic_df(s:str):
    """
    Loads a demographic dataset from a CSV file.

    Parameters:
    -----------
    s : str
        The path of the CSV file to load.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the demographic data from the specified CSV file.
    """
    return pd.read_csv(s)



def age_analysis(csv_file):
    """
    Analyzes age demographics from a CSV file and prints a summary.

    This function reads age data, categorizes participants into age brackets,
    and calculates the percentage of participants in each bracket.

    Parameters:
    -----------
    csv_file : str, optional
        The path of the CSV file containing age data (default is 'age.csv').

    Returns:
    --------
    None
        The function prints a formatted summary of age demographics with percentages
        for each age bracket (18-25, 26-35, 36-45, and over 45 years).

    Notes:
    ------
    - Non-numeric age values (like 'CONSENT_REVOKED') are removed from the analysis.
    - Percentages are rounded to the nearest integer.
    """
    df = pd.read_csv(csv_file)
    # Clean the data - remove non-numeric age values like 'CONSENT_REVOKED'
    df = df[pd.to_numeric(df['Age'], errors='coerce').notna()]
    df['Age'] = pd.to_numeric(df['Age'])

    # Define age brackets
    age_brackets = {
        '18-25': (18, 25),
        '26-35': (26, 35),
        '36-45': (36, 45),
        'over 45': (46, float('inf'))
    }

    # Count participants in each age bracket
    total_participants = len(df)
    bracket_counts = {}

    for bracket, (min_age, max_age) in age_brackets.items():
        count = len(df[(df['Age'] >= min_age) & (df['Age'] <= max_age)])
        percentage = (count / total_participants) * 100
        bracket_counts[bracket] = round(percentage)

    out = f'''Age demographics indicated that:\n
        - {bracket_counts['18-25']}% of participants were between 18-25 years
        - {bracket_counts['26-35']}% between 26-35 years
        - {bracket_counts['36-45']}% between 36-45 years
        - {bracket_counts['over 45']}% over 45 years of age.
    '''
    print(out)



def ethnicity_analysis(csv_file):
    """
    Analyzes ethnicity demographics from a CSV file and prints a summary.

    This function reads ethnicity data, calculates the percentage of participants
    from each ethnic background, and presents the results in a formatted output.

    Parameters:
    -----------
    csv_file : str, optional
        The path of the CSV file containing ethnicity data (default is 'ethnicity.csv').

    Returns:
    --------
    None
        The function prints a formatted summary of ethnicity demographics with percentages
        for each ethnic group (White/Caucasian, Asian, Black/African, Hispanic/Latino,
        and multiracial or other).
    """
    df = pd.read_csv(csv_file)
    # Convert count to numeric
    df['count'] = pd.to_numeric(df['count'])

    # Calculate total participants (excluding CONSENT_REVOKED)
    valid_participants = df[df['Ethnicity simplified'] != 'CONSENT_REVOKED']['count'].sum()

    # Calculate percentages
    df['percentage'] = df['count'] / valid_participants * 100

    # Create the percentages in the desired order from the example
    percentages = []
    for ethnicity, output_name in [
        ('White', 'White/Caucasian'),
        ('Asian', 'Asian'),
        ('Black', 'Black/African'),
        ('Hispanic', 'Hispanic/Latino'),
        ('Mixed', 'multiracial or other')
    ]:
        if ethnicity in df['Ethnicity simplified'].values:
            pct = round(df[df['Ethnicity simplified'] == ethnicity]['percentage'].iloc[0])
            percentages.append(f"{pct}% as {output_name}")

    out = f'''The study included participants from various ethnic backgrounds, with:\n
    - {"\n\t- ".join(percentages)}'''
    print(out)



def language_analysis(csv_file):
    """
    Analyzes language proficiency demographics from a CSV file and prints a summary.

    This function reads language data and presents the five most prevalent languages
    among participants with their respective percentages.

    Parameters:
    -----------
    csv_file : str, optional
        The path of the CSV file containing language data (default is 'language.csv').

    Returns:
    --------
    None
        The function prints a formatted summary of the five most prevalent languages
        among participants with percentages rounded to three decimal places.

    """
    df = pd.read_csv(csv_file)

    out = f"""
    The research study encompassed participants proficient in various languages. 
    The five most prevalent languages among the participants were as follows:\n
        - {'\n\t\t- '.join([f'{row['Language']} = {round(row['percentage'], 3)}%' for _, row in df.head().iterrows()])}
    """
    print(out)



# Create a mapping of countries to continents
continent_mapping = {
    # Africa
    'South Africa': 'Africa',
    'Nigeria': 'Africa',
    'Zimbabwe': 'Africa',
    'Kenya': 'Africa',
    'Ghana': 'Africa',

    # Europe
    'United Kingdom': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Spain': 'Europe',
    'Hungary': 'Europe',
    'Italy': 'Europe',
    'Greece': 'Europe',
    'Ireland': 'Europe',
    'Germany': 'Europe',
    'Slovenia': 'Europe',
    'Croatia': 'Europe',
    'Czech Republic': 'Europe',
    'Estonia': 'Europe',
    'Austria': 'Europe',
    'Slovakia': 'Europe',
    'Belarus': 'Europe',
    'Romania': 'Europe',
    'Bulgaria': 'Europe',
    'Switzerland': 'Europe',
    'Serbia': 'Europe',
    'France': 'Europe',
    'Russian Federation': 'Europe',

    # North America
    'United States': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America',
    'American Samoa': 'Oceania',  # Actually Oceania, not North America

    # South America
    'Chile': 'South America',
    'Brazil': 'South America',

    # Asia
    'Turkey': 'Asia',
    'Philippines': 'Asia',
    'India': 'Asia',
    'Korea': 'Asia',
    'Vietnam': 'Asia',
    'Cambodia': 'Asia',
    'China': 'Asia',
    'Pakistan': 'Asia',
    'Bangladesh': 'Asia',

    # Other/Unknown
    'CONSENT_REVOKED': 'Unknown'
}


def nationality_analysis(csv_file):
    """
    Analyzes language proficiency demographics from a CSV file and prints a summary.

    This function reads language data and presents the five most prevalent languages
    among participants with their respective percentages.

    Parameters:
    -----------
    csv_file : str, optional
        The path of the CSV file containing language data (default is 'language.csv').

    Returns:
    --------
    None
        The function prints a formatted summary of the five most prevalent languages
        among participants with percentages rounded to three decimal places.
    """
    df = pd.read_csv(csv_file)
    df['continent'] = df['Nationality'].map(continent_mapping)

    # Check if any countries don't have a mapping
    missing_mappings = df[df['continent'].isna()]['Nationality'].unique()
    if len(missing_mappings) > 0:
        print(f"Warning: Missing continent mappings for: {', '.join(missing_mappings)}")
        # For the purpose of this analysis, let's categorize unknown as 'Unknown'
        df['continent'].fillna('Unknown', inplace=True)

    # Calculate the total number of participants
    total_count = df['count'].sum()

    # Group by continent and calculate percentages
    continent_distribution = df.groupby('continent')['count'].sum()
    continent_percentages = (continent_distribution / total_count * 100).round(0).astype(int)

    # Sort continents by count in descending order
    continent_percentages = continent_percentages.sort_values(ascending=False)

    # Generate the output string
    output_parts = []
    for continent, percentage in continent_percentages.items():
        if continent != 'Unknown':  # Skip unknown in the output string
            output_parts.append(f"{continent} ({percentage}%)")

    out = f"""
    The study participants exhibited a diverse geographical distribution. 
    The breakdown of participants by continent was as follows:\n
    - {"\n\t- ".join(output_parts)}
    """
    print(out)

    return continent_percentages, continent_distribution