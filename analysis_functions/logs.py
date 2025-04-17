"""
User Study Log Analysis Framework

This module provides a structured framework for analyzing log data from user studies.
It enables loading, processing, and querying chronological event data for multiple users.

BEFORE STARTING ANALYSIS:
1. Edit questionnaire_log_name to match the log name used for questionnaires in your experiment.
2. Edit the method process_answer according to the structure of questionnaire data in your experiment.
3. Edit the environment variables in the .env file.
4. Edit the method study_started to match the log structure.

USAGE PATTERN:
1. Data Loading: ALWAYS use the Logs class as the entry point for retrieving data.
   The Logs class handles data loading, preprocessing, and timeline creation.

2. Data Analysis: Once you have a Timeline object from Logs, ONLY use Timeline methods
   for analysis. Do not attempt to modify or directly access Timeline internals.

Example:
    # Load data
    study_logs = Logs(table="user_study_results")

    # Iterate through all user timelines
    for user_id, timeline in study_logs.items():
        print(user_id)
        print(timeline)

    # Access a specific user's timeline
    user_timeline = study_logs.logs["user123"]

    # Analyze using Timeline methods
    all_logs = user_timeline.get_logs()
    answers = user_timeline.get_answers("satisfaction_survey")

    # To process all timelines in parallel, use process_in_parallel method
"""

import json
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Any, Callable

import pandas as pd


"""
    Edit the following as instructed above
"""
# Edit this to match the log name used for questionnaires in your experiment
questionnaire_log_name = [
    "tia_post_task",
    "desirability_of_control",
    "freeTextAnswers",
    "tia_pre_task",
    "grid_questionnaire",
    "tlx_questionnaire"
]


"""
    This method should return true iff the log marks the start of the study, false otherwise
"""
def study_started(log):
    return log["type"] == "study_started"

def process_answer(timeline, log_data, test_name, timestamp):
    """
    Process questionnaire data and add it to the timeline as an AnswersEvent.

    This function extracts answers from questionnaire log data and organizes them
    into an AnswersEvent object that gets added to the timeline. It handles the
    specific format of questionnaire data used in the experiment.

    Args:
        timeline (Timeline): The user timeline to add the answers to.
        log_data (Dict): The raw questionnaire data containing questions and answers.
        test_name (str): The base name of the questionnaire (e.g. 'answered_questions').
        timestamp (datetime): When the questionnaire was completed.

    Returns:
        None: The function modifies the timeline in-place by adding an AnswersEvent.
    """
    # The following is a prototypical implementation of process_answer.
    # You may modify it as you see fit according to the structure of your log data.

    # Create a unique identifier for this questionnaire by appending the experiment step
    # If there only is one step per user, test_name_full should be the same as test_name

    #test_name_full = f"{test_name}_{log_data['exp_step']}" # TODO: Old version
    test_name_full = f"{test_name}"


    # Only process this questionnaire if we haven't seen it before for this user
    if test_name_full not in timeline.test_names:
        # Create a new AnswersEvent for this questionnaire
        event = AnswersEvent(test_name_full, timestamp)

        # Add the event to the timeline
        timeline.events.append(event)

        # Store the index of this event for later reference
        timeline.test_names[test_name_full] = len(timeline.events) - 1


        # Process each question in the questionnaire data
        for question in log_data:
            # Skip attention checks. Here we assume they all start with "Please select" and that
            # no other questions start with that. Feel free to modify this if the wording is different in your case.
            if question.isnumeric() and "Please select " not in log_data[question]['text']:
                # Add this question-answer pair to the event
                event.add_answer(log_data[question]['text'], log_data[question]['answer'])
            # Add free-text answers as well
            if question == 'useful' or question == 'improvements':
                event.add_answer(question, log_data[question])




"""
    Do not edit below this line unless you would like to modify the framework
"""


class Questions:
    """
    A container class for managing questions across different tests.

    This class maintains a mapping of test names to questions and provides
    methods to add questions and retrieve their indices efficiently.

    Attributes:
        tests (dict): Dictionary mapping test names to lists of questions.
        _question_indices (defaultdict): Cache of question indices for faster lookups.
    """

    def __init__(self):
        self.tests = dict()  # Test_name => list of questions
        self._question_indices = defaultdict(dict)  # Test_name => {question => index}

    def add_question(self, test_name: str, question: str) -> int:
        """
        Add a question to a specific test and return its index.

        If the question already exists, its cached index is returned.

        Args:
            test_name: The name of the test this question belongs to.
            question: The text of the question.

        Returns:
            int: The index of the question within its test.
        """
        if test_name not in self.tests:
            self.tests[test_name] = []

        # Use cached index if available
        if question in self._question_indices[test_name]:
            return self._question_indices[test_name][question]

        # Add new question and cache its index
        if question not in self.tests[test_name]:
            self.tests[test_name].append(question)
        index = self.tests[test_name].index(question)
        self._question_indices[test_name][question] = index
        return index

    def __getitem__(self, test_name: str) -> List[str]:
        """
        Retrieve all questions for a specific test.

        Args:
            test_name: The name of the test to retrieve questions for.

        Returns:
            List[str]: A list of questions for the specified test.
        """
        return self.tests[test_name]


questions = Questions()


class Event:
    """
    Base class representing a generic event in the timeline.

    Attributes:
        event_name (str): Name of the event.
        timestamp (datetime): When the event occurred.
    """

    def __init__(self, event_name: str, timestamp: datetime):
        self.event_name = event_name
        self.timestamp = timestamp

    def __str__(self) -> str:
        """Return a string representation of the event."""
        return '[Generic event] ' + self.event_name


class LogEvent(Event):
    """
    Represents a logged event with associated data.

    This class extends Event to include structured log data and provides
    efficient access to this data through dictionary-like operations.

    Attributes:
        log_information (Dict): The raw log data associated with this event.
        _cache (Dict): Cache for computed values to improve performance.
    """

    def __init__(self, log_name: str, log_data: Dict, timestamp: datetime):
        super().__init__(log_name, timestamp)
        self.log_information = log_data
        self._cache = {}  # Cache for computed values

    def __getitem__(self, item: Any) -> Any:
        """
        Access log data by key or index with caching for performance.

        Args:
            item: The key or index to access in the log data.

        Returns:
            The value associated with the key/index, or None if not found.
        """
        if item in self._cache:
            return self._cache[item]

        result = None
        if isinstance(self.log_information, list):
            result = self.log_information[item]
        else:
            result = self.log_information.get(item)

        self._cache[item] = result
        return result

    def __str__(self) -> str:
        """Return a formatted string representation of the log event."""
        return f'[Log] {self.event_name} at {self.timestamp} - {str(self.log_information)}\n'


class AnswersEvent(Event):
    """
    Represents a set of answers to questions in a test.

    This class tracks answers by their question indices and provides
    efficient access to answers through question text lookup.

    Attributes:
        answers (dict): Dictionary mapping question indices to answers.
        _question_lookup (dict): Cache for question to index lookups.
    """

    def __init__(self, test_name: str, timestamp: datetime):
        super().__init__(test_name, timestamp)
        self.answers = dict()  # question_index => answer
        self._question_lookup = {}  # Cache for question lookups

    def add_answer(self, question: str, answer: Any) -> None:
        """
        Add an answer for a specific question.

        Args:
            question: The text of the question being answered.
            answer: The answer provided for the question.
        """
        index = questions.add_question(self.event_name, question)
        self.answers[index] = answer
        self._question_lookup[question] = index

    def __getitem__(self, item: str) -> Any:
        """
        Retrieve an answer by question text.

        Args:
            item: The text of the question to retrieve the answer for.

        Returns:
            The answer associated with the question.
        """
        if item in self._question_lookup:
            index = self._question_lookup[item]
        else:
            index = questions[self.event_name].index(item)
            self._question_lookup[item] = index
        return self.answers[index]

    def __str__(self) -> str:
        """Return a formatted string representation of all question/answer pairs."""
        tmpstr = ""
        for index, answer in self.answers.items():
            tmpstr += f"{questions[self.event_name][index]} - {str(answer)}\n"
        return f'[Answers] {self.event_name}\n{tmpstr}'
        # return f'[Answers] {self.event_name} - {questions[self.event_name]}'


    def get_questions(self) -> List[str]:
        # return questions[self.event_name]
        return [questions[self.event_name][index] for index, _ in self.answers.items()]


class Timeline:
    """
    Represents a chronological sequence of events for a specific user.

    This class provides methods to query events by various criteria and
    efficiently access logs and answers through indexing and caching.

    IMPORTANT: During analysis, ONLY use the methods of this class.
    Do not attempt to modify Timeline objects directly or access their internal
    structure. Always obtain Timeline objects through the Logs class, never
    create them manually.

    Attributes:
        user (str): The identifier for the user this timeline belongs to.
        events (List[Event]): Chronologically ordered list of events.
        test_names (Dict[str, int]): Mapping of test names to event indices.
        editing (bool): Whether the timeline is currently being edited.
        _log_event_index (Dict): Index of log events by name for faster lookups.
        _answer_event_index (Dict): Index of answer events by test name.
        _event_type_index (Dict): Index of events by their type.
        _cached_logs (Dict): Cache for expensive log query results.
        _cached_answers (Dict): Cache for expensive answer query results.
    """

    def __init__(self, user: str):
        self.user = user
        self.events: List[Event] = []
        self.test_names: Dict[str, int] = {}  # Test_name => index in events
        self.editing = True

        # Indexes for faster lookups
        self._log_event_index: Dict[str, List[int]] = defaultdict(list)  # log_name => list of indices
        self._answer_event_index: Dict[str, int] = {}  # test_name => index
        self._event_type_index: Dict[type, List[int]] = defaultdict(list)  # event_type => list of indices

        # Cache for expensive operations
        self._cached_logs = {}
        self._cached_answers = {}

    def _rebuild_indices(self) -> None:
        """
        Rebuild all internal indices after timeline modifications.

        This method is called after timeline editing is completed to ensure
        all lookup structures are up to date.
        """
        self._log_event_index.clear()
        self._answer_event_index.clear()
        self._event_type_index.clear()

        for idx, event in enumerate(self.events):
            if isinstance(event, LogEvent):
                self._log_event_index[event.event_name].append(idx)
                self._event_type_index[LogEvent].append(idx)
            elif isinstance(event, AnswersEvent):
                self._answer_event_index[event.event_name] = idx
                self._event_type_index[AnswersEvent].append(idx)

    def edit(self) -> None:
        """
        Enter editing mode for the timeline.

        This method allows modifications to the timeline and clears any cached
        results that might become invalid due to the changes.
        """
        self.editing = True
        # Clear caches when editing starts
        self._cached_logs.clear()
        self._cached_answers.clear()

    def close(self) -> None:
        """
        Close the timeline for editing and prepare it for querying.

        This method sorts events chronologically and rebuilds all indices
        to ensure efficient lookup operations.
        """
        self.events.sort(key=lambda x: x.timestamp)
        self._rebuild_indices()
        self.editing = False

    def check_access(self) -> None:
        """
        Verify that the timeline is not in editing mode before performing queries.

        Raises:
            RuntimeError: If the timeline is still in editing mode.
        """
        if self.editing:
            raise RuntimeError('In order to access a Timeline, you need to close it first: timeline.close().')

    def check_edit(self) -> None:
        """
        Verify that the timeline is in editing mode before modifications.

        Raises:
            RuntimeError: If the timeline is not in editing mode.
        """
        if not self.editing:
            raise RuntimeError('In order to edit a Timeline, you need to call timeline.edit() first.')

    def add_event(self, test_name: str, log_data: Dict, timestamp: datetime) -> None:
        """
        Add a new event to the timeline.

        This method handles both regular log events and special answer events,
        processing them appropriately based on their type.

        Args:
            test_name: The name of the test or log event.
            log_data: The data associated with the event.
            timestamp: When the event occurred.

        Raises:
            RuntimeError: If the timeline is not in editing mode.
        """
        self.check_edit()

        if test_name in questionnaire_log_name:
            process_answer(self, log_data, test_name, timestamp)
        else:
            self.events.append(LogEvent(log_data.get('type'), log_data, timestamp))

    @lru_cache(maxsize=1024)
    def get_logs(self, log_name: Optional[str] = None, precise: bool = True) -> List[LogEvent]:
        """
        Retrieve log events matching the specified criteria.

        This method supports both exact and partial matching of log names
        and utilizes caching for improved performance.

        Args:
            log_name: Optional name of logs to retrieve. If None, all logs are returned.
            precise: If True, only exact matches are returned. If False, partial matches are included.

        Returns:
            List[LogEvent]: A list of matching log events.

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        self.check_access()

        cache_key = (log_name, precise)
        if cache_key in self._cached_logs:
            return self._cached_logs[cache_key]

        if log_name is None:
            result = [self.events[idx] for idx in self._event_type_index[LogEvent]]
        else:
            indices = []
            if precise:
                indices = self._log_event_index.get(log_name, [])
            else:
                for name, idx_list in self._log_event_index.items():
                    if log_name in name:
                        indices.extend(idx_list)
            result = [self.events[idx] for idx in indices]

        self._cached_logs[cache_key] = result
        return result

    def get_logs_by_custom_function(self, fct: Callable[[LogEvent], bool]) -> List[LogEvent]:
        """
        Retrieve log events that satisfy a custom filtering function.

        Args:
            fct: A function that takes a LogEvent and returns True if it should be included.

        Returns:
            List[LogEvent]: A list of log events that satisfy the filtering function.

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        self.check_access()
        return [event for event in self.get_logs() if fct(event)]

    def get_events(self, event_name: Optional[str] = None, precise: bool = True, force_edit: bool = False) -> List[Event]:
        """
        Retrieve events matching the specified criteria.

        Args:
            event_name: Optional name of events to retrieve. If None, all events are returned.
            precise: If True, only exact matches are returned. If False, partial matches are included.
            force_edit: If True, force self.check_access()

        Returns:
            List[Event]: A list of matching events.

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        if not force_edit:
            self.check_access()

        if event_name is None:
            return self.events

        return [event for event in self.events if
                (precise and event.event_name == event_name) or
                (not precise and event_name in event.event_name)]

    def get_answers(self, test_name: Optional[str] = None) -> List[AnswersEvent]:
        """
        Retrieve answer events for the specified test.

        Args:
            test_name: Optional name of the test to retrieve answers for.
                      If None, all answer events are returned.

        Returns:
            List[AnswersEvent]: A list of matching answer events.

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        self.check_access()

        if test_name in self._cached_answers:
            return self._cached_answers[test_name]

        if test_name is None:
            result = [self.events[idx] for idx in self._event_type_index[AnswersEvent]]
        else:
            idx = self._answer_event_index.get(test_name)
            result = [self.events[idx]] if idx is not None else []

        self._cached_answers[test_name] = result
        return result

    def get_latest_log_before_event(self, log_name: str, event: Event, precise: bool = False) -> LogEvent:
        """
        Find the most recent log event of a specific type that occurred before the given event.

        Args:
            log_name: The name of the log events to search for.
            event: The reference event to search before.
            precise: If True, only exact matches of log_name are considered.
                    If False, partial matches are included.

        Returns:
            LogEvent: The most recent matching log event before the reference event.

        Raises:
            RuntimeError: If the timeline is in editing mode.
            FileNotFoundError: If no matching log event is found.
        """
        self.check_access()
        event_index = self.events.index(event)

        relevant_logs = self.get_logs(log_name, precise)
        for log in reversed(relevant_logs):
            if self.events.index(log) < event_index:
                return log

        raise FileNotFoundError(f'No {log_name} log was found prior to the event {str(event)}')

    def get_next_log_after_event(self, log_name: str, event: Event, precise: bool = False) -> LogEvent:
        """
        Find the next log event of a specific type that occurred after the given event.

        Args:
            log_name: The name of the log events to search for.
            event: The reference event to search after.
            precise: If True, only exact matches of log_name are considered.
                    If False, partial matches are included.

        Returns:
            LogEvent: The next matching log event after the reference event.

        Raises:
            RuntimeError: If the timeline is in editing mode.
            FileNotFoundError: If no matching log event is found.
        """
        self.check_access()
        event_index = self.events.index(event)

        relevant_logs = self.get_logs(log_name, precise)
        for log in relevant_logs:
            if self.events.index(log) > event_index:
                return log

        raise FileNotFoundError(f'No {log_name} log was found after the event {str(event)}')

    def get_events_between(self, start_event: Event, end_event: Event,
                           event_name: Optional[str] = None) -> List[Event]:
        """
        Retrieve events that occurred between two reference events.

        Args:
            start_event: The event marking the beginning of the time window (exclusive).
            end_event: The event marking the end of the time window (exclusive).
            event_name: Optional name to filter the events by.

        Returns:
            List[Event]: Events that occurred between start_event and end_event.

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        self.check_access()
        start_idx = self.events.index(start_event) + 1
        end_idx = self.events.index(end_event)

        events = self.events[start_idx:end_idx]
        if event_name is None:
            return events
        return [event for event in events if event.event_name == event_name]


    def get_events_by_names(self, event_names: Optional[List[str]] = None,
                            start_event: Event = None, end_event: Event = None) -> List[Event]:
        """
        Retrieve events that occurred between two reference events.

        Args:
            start_event: The event marking the beginning of the time window (exclusive).
            end_event: The event marking the end of the time window (exclusive).
            event_names: Optional names to filter the events by.

        Returns:
            List[Event]: Events that occurred between start_event and end_event.

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        self.check_access()
        if start_event is not None and end_event is not None:
            start_idx = self.events.index(start_event) + 1
            end_idx = self.events.index(end_event)
            events = self.events[start_idx:end_idx]
        else:
            events = self.events

        if event_names is None:
            return events
        return [event for event in events if event.event_name in event_names]


    @lru_cache(maxsize=1024)
    def get_partition(self, partition_name: str, threshold: float,
                      system: Optional[str] = None) -> str:
        """
        Categorize a user based on their answer to a specific question.

        This method determines if a user's response is above or below a threshold
        and returns an appropriate category label.

        Args:
            partition_name: The name of the test containing the partitioning question.
            threshold: The threshold value for categorization.
            system: Optional system name if the answer is nested within a system key.

        Returns:
            str: Category label ('high_X' or 'low_X' where X is the partition_name).

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        self.check_access()
        data = self.get_answers(partition_name)[0]
        if system is not None:
            data = data[system]
        else:
            data = data.answers[0]
        return f'high_{partition_name}' if data >= threshold else f'low_{partition_name}'

    def __str__(self) -> str:
        """
        Return a formatted string representation of the entire timeline.

        Raises:
            RuntimeError: If the timeline is in editing mode.
        """
        self.check_access()
        return f"TIMELINE FOR USER {self.user}\n" + \
            "".join(f"{str(event)}----------\n" for event in self.events)


class Logs:
    """
    A container class for managing multiple user timelines.

    This class handles the loading and preprocessing of log data from
    various sources and provides access to individual user timelines.

    IMPORTANT: This is the ONLY class that should be used directly for data retrieval.
    Always initialize a Logs object to get access to Timeline objects. Do not attempt
    to create Timeline objects manually.

    Attributes:
        logs (Dict[str, Timeline]): Dictionary mapping user IDs to their timelines.
        userid_prolific_mapping (Dict[str, str]): Dictionary mapping user IDs to their Prolific IDs.
        _cache (Dict): Cache for expensive operations.
    """

    def __init__(self, table: Optional[str] = None,
                 users_table: str = "user_model",
                 preprocessed_data: Optional[Dict[str, Timeline]] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 users_dataframe: Optional[pd.DataFrame] = None,
                 force_reload: bool = False,
                 to_csv: bool = False,):
        """
        Initialize the logs container with data from the specified source.

        Args:
            table: Optional name of the database table to load data from.
            users_table: Name of the database table containing user data to match user IDs with prolific.
            preprocessed_data: Optional preprocessed timeline data to use directly.
            dataframe: Optional pandas DataFrame containing log data.
            users_dataframe: Optional pandas DataFrame containing user data to match user IDs with prolific.
            force_reload: Whether to force reloading data from the source
                         instead of using cached data.
            to_csv: Whether to save data into a CSV file.
        """
        print('Init log')
        self.logs: Dict[str, Timeline] = {}
        self._cache = {}

        if preprocessed_data is not None:
            self.logs = preprocessed_data
        else:
            results = dataframe

            for _, row in results.iterrows():
                log_data = json.loads(row['log_data'])
                if type(row['timestamp']) is str:
                    row['timestamp'] = pd.to_datetime(row['timestamp'])
                if study_started(log_data) or row['user_id'] not in self.logs:
                    # Re-initialize the timeline if the study was started again, or if it was not initialized before.
                    self.logs[row['user_id']] = Timeline(row['user_id'])
                self.logs[row['user_id']].add_event(
                    log_data['type'], log_data, row['timestamp'])

        self.userid_prolific_mapping = {}

        users_df = users_dataframe
        for _, row in users_df.iterrows():
            self.userid_prolific_mapping[row['user_id']] = row['prolific_id']

        for timeline in self.logs.values():
            timeline.close()

        print('Data loaded successfully!')

    def get_prolific_id(self, user_id: str) -> Any | None:
        """
        Retrieve the Prolific ID associated with a specific user.

        Args:
            user_id: The identifier of the user to retrieve the Prolific ID for.

        Returns:
            Any: The Prolific ID associated with the specified user, or None if not found.
        """
        return self.userid_prolific_mapping.get(user_id, None)


    def items(self):
        """
        Return an iterator over (user_id, timeline) pairs.

        This is the recommended way to access all timelines for analysis.

        Returns:
            Iterator yielding (str, Timeline) pairs where each Timeline
            is properly initialized and ready for analysis.
        """
        return self.logs.items()

    def get_timeline(self, user_id: str) -> Timeline:
        """
        Retrieve a specific user's timeline.

        This is the recommended way to access a single timeline for analysis.

        Args:
            user_id: The identifier of the user whose timeline to retrieve.

        Returns:
            Timeline: The timeline for the specified user.

        Raises:
            KeyError: If no timeline exists for the specified user.
        """
        if user_id not in self.logs:
            raise KeyError(f"No timeline exists for user '{user_id}'")
        return self.logs[user_id]

    def __str__(self) -> str:
        """Return a formatted string representation of all user timelines."""
        return "".join(f"LOGS FOR USER {user}\n{timeline}\n\n"
                       for user, timeline in self.logs.items())

    def process_in_parallel(self, process_function, merge_function=None, n_processes=None,
                            show_progress=True, **kwargs):
        """
        Process all timelines in parallel using multiprocessing.

        This method allows efficient parallel processing of user timelines
        using a specified processing function. Results can optionally be
        merged using a provided merge function.

        Args:
            process_function (Callable): A function that takes a tuple of (user_id, timeline)
                                        and any additional kwargs, and returns a result.
            merge_function (Callable, optional): A function that takes (result, aggregated_results)
                                               and merges the result into the aggregated results.
                                               If None, results are returned as a list.
            n_processes (int, optional): Number of processes to use. If None, will use
                                        cpu_count() - 1 or 1, whichever is greater.
            show_progress (bool): Whether to show a progress bar (requires tqdm).
            **kwargs: Additional keyword arguments to pass to process_function.

        Returns:
            Any: If merge_function is provided, returns the merged results.
                 Otherwise, returns a list of individual results.

        Example:
            def process_single_user(user_data, threshold=0.5):
                user_id, timeline = user_data
                # Process timeline data
                result = {'user_id': user_id, 'metrics': calculate_metrics(timeline)}
                return result

            def merge_results(result, all_metrics, all_questionnaire_data):
                if result is not None:
                    all_metrics[result['user_id']] = result['metrics']

            # Initialize result containers
            all_metrics = {}
            all_questionnaire_data = {}

            # Process in parallel and merge results
            logs.process_in_parallel(
                process_single_user,
                merge_function=lambda r, *args: merge_results(r, all_metrics, all_questionnaire_data),
                threshold=0.7
            )
        """
        import multiprocessing
        import os

        # Ensure we have the required packages
        try:
            from tqdm import tqdm
        except ImportError:
            if show_progress:
                print("Warning: tqdm package not found. Progress bar will not be shown.")
                show_progress = False

        # Determine number of processes
        if n_processes is None:
            n_processes = max(1, multiprocessing.cpu_count() - 1)

        # Prepare arguments list
        args_list = list(self.logs.items())

        # Create a wrapper function to handle kwargs
        def process_wrapper(user_data):
            return process_function(user_data, **kwargs)

        with multiprocessing.Pool(processes=n_processes) as pool:
            # Process items in parallel with optional progress bar
            if show_progress:
                from tqdm import tqdm
                results_iter = pool.imap_unordered(process_wrapper, args_list)
                results = list(tqdm(
                    results_iter,
                    total=len(args_list),
                    desc="Processing users",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
                ))
            else:
                results = pool.map(process_wrapper, args_list)

        # If a merge function is provided, merge the results
        if merge_function is not None:
            aggregated_result = None
            for result in results:
                if result is not None:
                    aggregated_result = merge_function(result, aggregated_result)
            return aggregated_result

        # Otherwise, return the list of results
        return [r for r in results if r is not None]