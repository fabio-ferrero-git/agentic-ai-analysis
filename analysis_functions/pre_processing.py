import json
import pandas as pd

from analysis_functions.logs import Logs
import analysis_functions.utils as u
import os

# System 0, System 1, System 2
_N_SYSTEMS = 3
_PATH = './data/pre_processed/'
FILES = [
    'study_sample.json',
    'study_durations.json',
    'questionnaires_durations.json',
    'questionnaires_results.json',
    'free_text_answer.json',
    'n_actions_in_5min.json',
    'n_replied.json',
    'n_replied_full.json',
    'users_replies.json',
]


def study_questions():
    """
    Load and return study questions from a JSON file.

    Returns:
        dict: The study questions data loaded from the JSON file.
    """
    file = open('./data/study_question.json', 'r')
    s = json.loads(file.read())
    file.close()
    return s



def ensure_directory_exists(path):
    """
    Create the directory if it doesn't exist.

    Args:
        path (str): The directory path to check or create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")



def pre_process():
    """
    Preprocess study data and store the results in JSON files.

    This function either loads existing preprocessed data from JSON files
    or creates new preprocessed data by analyzing study logs, questionnaires,
    and user interactions. The preprocessed data includes information about
    study samples, durations, questionnaire results, user actions, and more.

    Returns:
        dict: A dictionary containing various preprocessed datasets:
            - study_sample: Count of participants per system
            - study_durations: Study duration times per system
            - questionnaires_durations: Time spent on questionnaires per system
            - questionnaires_results: Results of questionnaires per system
            - free_text_answer: Positive and negative text answers per system
            - n_actions_in_5min: Number of user actions per system
            - n_replied: Ratio of replied messages per system
            - n_replied_full: Ratio considering follow-ups per system
            - users_replies: Count of replies per user per system
    """
    ensure_directory_exists(_PATH)
    all_files_exist = all(os.path.exists(_PATH + file) for file in FILES)

    if all_files_exist:
        print('Load preprocessed data from JSON files...')
        # Load preprocessed data from JSON files
        preprocessed_data = {}
        for file in FILES:
            with open(_PATH + file, 'r') as f:
                data_dict = json.load(f)
                preprocessed_data[file[:-5]] = {int(k) : v for k, v in data_dict.items()}

        print('Data loaded successfully!')
        return preprocessed_data

    # If preprocessed data doesn't exist, proceed with preprocessing
    print('Preprocessing data...')
    user_model = pd.read_csv('./data/csv_df_user_model.csv')
    study_logs = Logs(dataframe=pd.read_csv('./data/csv_df_test_logs.csv'), users_dataframe=user_model)

    file = open('./data/scenarios.json', 'r')
    scenarios = json.loads(file.read())
    file.close()

    study_sample = {c : 0 for c in range(_N_SYSTEMS)}
    study_durations = {c : [] for c in range(_N_SYSTEMS)}
    questionnaires_durations = {c: {} for c in range(_N_SYSTEMS)}
    questionnaires_results = {c : {} for c in range(_N_SYSTEMS)}
    free_text_answer = {c : {'positive': [], 'negative' : []} for c in range(_N_SYSTEMS)}
    n_actions_in_5min = {c : [] for c in range(_N_SYSTEMS)}
    n_replied = {c : [] for c in range(_N_SYSTEMS)}
    n_replied_full = {c : [] for c in range(_N_SYSTEMS)}
    users_replies = {c : [] for c in range(_N_SYSTEMS)}

    for user_id, timeline in study_logs.items():

        condition = u.get_user_condition(user_model, user_id)
        job, scenario_idx = u.get_user_scenario(user_model, user_id)
        scenario = scenarios[job][scenario_idx]
        emails_with_followup = len([email['id'] for email in scenario.get('baseEmails') if len(email.get('followUps')) > 0])

        # Get useful logs
        start_logs = timeline.get_logs(u.log_start)
        end_logs = timeline.get_logs(u.log_end)
        phases_logs = timeline.get_events(u.log_phase)

        start_interface_log = [e for e in phases_logs if e['to'] == 'interface'][0]
        end_interface_log = [e for e in phases_logs if e['from'] == 'interface'][0]

        study_durations[condition].append(u.timestamp_diff(end_logs[0], start_logs[0]))

        for q, duration in u.get_questionnaires_duration(timeline, phases_logs).items():
            questionnaires_durations[condition].setdefault(q, []).append(duration)

        for q, result in u.get_questionnaires_results(timeline.get_answers()).items():
            questionnaires_results[condition].setdefault(q, []).append(result)

        # Correct tia-pre-task (3 quests) and desirability_of_control
        u.fix_pre_task_questionnaire(questionnaires_results, condition)

        pos, neg = u.get_free_text_answers(timeline.get_answers('freeTextAnswers'))
        free_text_answer[condition]['positive'] += [(user_id, pos)] if pos else []
        free_text_answer[condition]['negative'] += [(user_id, neg)] if neg else []

        new_message = timeline.get_events_between(start_interface_log, end_interface_log, 'new_message')
        message_replied = timeline.get_events_between(start_interface_log, end_interface_log, 'message_replied')

        actions_names = ['event_created', 'event_updated', 'event_deleted', 'todo_created', 'todo_updated']
        user_actions = timeline.get_events_by_names(actions_names)

        n_actions_in_5min[condition].append(len(user_actions) + len(message_replied))

        n_replied[condition].append(
            len({x.log_information['msg']['id'] for x in message_replied}) /
            len({x.log_information['id'] for x in new_message})
        )

        # Considering follow-ups
        n_replied_full[condition].append(
            len({x.log_information['msg']['id'] for x in message_replied}) /
            (len({x.log_information['id'] for x in new_message}) + emails_with_followup)
        )

        # Number of messages replied per user
        users_replies[condition].append(len(message_replied))

    for s in study_sample:
        study_sample[s] = len(study_durations[s])

    preprocessed_data = {
        'study_sample' : study_sample,
        'study_durations': study_durations,
        'questionnaires_durations': questionnaires_durations,
        'questionnaires_results': questionnaires_results,
        'free_text_answer': free_text_answer,
        'n_actions_in_5min': n_actions_in_5min,
        'n_replied': n_replied,
        'n_replied_full': n_replied_full,
        'users_replies' : users_replies,
    }

    for key, value in preprocessed_data.items():
        _to_json(value, f"{key}.json")

    return preprocessed_data



def _to_json(data, filename, path=_PATH) -> None:
    """
    Save data as a JSON file.

    Args:
        data: The data to save.
        filename (str): The name of the JSON file.
        path (str, optional): The directory path to save the file. Defaults to _PATH.
    """
    ensure_directory_exists(path)
    with open(path + filename, "w") as out_file:
        json.dump(data, out_file)