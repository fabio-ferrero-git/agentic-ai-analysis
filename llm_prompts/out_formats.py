"""
This section defines the data models and JSON schema for the /ask-llm API.

The models include:
- AskLLMAction: Represents individual actions in the /ask-llm format.
- SuggestedActionModel: Represents suggested actions for the /ask-llm format.
- AskLLMFormat: Represents the overall JSON format for /ask-llm.
- UpdateAskLLMFormat: Represents the JSON format for updating actions in /ask-llm.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict



class AskLLMAction(BaseModel):
    """
    Represents an action for the AskLLM format.

    Attributes:
        start_action (bool): Indicates if the action is the start of a sequence.
        action (Literal): Specifies the type of action (e.g., 'create_draft', 'create_event', etc.).
        id (Optional[str]): Optional identifier for the action.
        draft (Optional[str]): Optional draft content.
        title (str): Title of the action.
        description (Optional[str]): Optional description of the action.
        datetime (Optional[str]): Optional datetime in ISO 8601 format.
        duration (Optional[str]): Duration of the action, default is "1".
        conflict (Optional[bool]): Indicates if there is a conflict.
        provisional (Optional[bool]): Indicates if the action is provisional.
        end_action (bool): Indicates if the action is the end of a sequence.
    """
    start_action: bool
    action: Literal['create_draft', 'create_event', 'edit_event',
    'delete_event', 'create_todo', 'edit_todo', 'mark_todo']
    id: Optional[str] = None
    draft: Optional[str] = None
    title: str
    description: Optional[str] = None
    datetime: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$')
    duration: Optional[str] = Field("1")
    conflict: Optional[bool] = None
    provisional: Optional[bool] = None
    end_action: bool



class SuggestedActionModel(BaseModel):
    """
    Represents a model for suggested actions.

    Attributes:
        model_config (ConfigDict): Configuration to forbid extra fields.
        start (bool): Indicates the start of a sequence.
        items (List[Literal]): List of suggested action types.
        end (bool): Indicates the end of a sequence.
    """
    model_config = ConfigDict(extra='forbid')
    start: bool
    items: List[Literal[
        'create_draft', 'create_event', 'edit_event',
        'delete_event', 'create_todo', 'edit_todo', 'mark_todo'
    ]]
    end: bool



class AskLLMFormat(BaseModel):
    """
    Represents the JSON format for AskLLM.

    Attributes:
        model_config (ConfigDict): Configuration to forbid extra fields.
        start (bool): Indicates the start of a sequence.
        items (List[Literal]): List of action types.
        end (bool): Indicates the end of a sequence.
        actions (List[AskLLMAction]): List of actions in the format.
    """
    model_config = ConfigDict(extra='forbid')
    start: bool
    items: List[Literal[
        'create_draft', 'create_event', 'edit_event',
        'delete_event', 'create_todo', 'edit_todo', 'mark_todo'
    ]]
    end: bool
    actions: List[AskLLMAction]



class UpdateAskLLMFormat(BaseModel):
    """
    Represents the JSON format for updating AskLLM actions.

    Attributes:
        action (Literal): Specifies the type of action (e.g., 'create_event', 'edit_event', etc.).
        id (Optional[str]): Optional identifier for the action.
        title (str): Title of the action.
        description (Optional[str]): Optional description of the action.
        datetime (Optional[str]): Optional datetime in ISO 8601 format.
        duration (Optional[str]): Duration of the action, default is "1".
        provisional (Optional[bool]): Indicates if the action is provisional.
    """
    action: Literal['create_event', 'edit_event', 'delete_event']
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    datetime: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$')
    duration: Optional[str] = Field("1")
    provisional: Optional[bool] = None



class FollowUpFormat(BaseModel):
    """
    Represents the JSON format for the follow-up generator.

    Attributes:
        id (str): Identifier for the follow-up.
        response (str): Response content.
        event_original_date (Optional[str]): Original event date in ISO 8601 format.
        event_new_date (Optional[str]): New event date in ISO 8601 format.
        day_difference (Optional[str]): Difference in days between the original and new dates.
        time_difference (Optional[str]): Difference in time between the original and new dates.
    """
    id: str
    response: str
    event_original_date: Optional[str] = Field(
        default=None,
        pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$'
    )
    event_new_date: Optional[str] = Field(
        default=None,
        pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$'
    )
    day_difference: Optional[str] = None
    time_difference: Optional[str] = None



# Dictionary containing JSON schemas for different output formats
output_format = {
    'ask-llm': AskLLMFormat.model_json_schema(),
    'follow-up-generator': FollowUpFormat.model_json_schema(),
    'update-ask-llm': UpdateAskLLMFormat.model_json_schema()
}



def clean_output_content(s: str, tokens: List[str]) -> str:
    """
    Cleans the output content by removing specified tokens.

    Args:
        s (str): The string to clean.
        tokens (List[str]): List of tokens to remove from the string.

    Returns:
        str: The cleaned string.
    """
    for token in tokens:
        s = s.replace(token, '')
    return s



def get_message(json_obj: dict) -> str:
    """
    Extracts the message content from a JSON object.

    Args:
        json_obj (dict): The JSON object containing the message.

    Returns:
        str: The extracted message content.
    """
    return json_obj['choices'][0]['message']['content']



def get_actions(json_obj: dict) -> any:
    """
    Extracts the actions from a JSON object.

    Args:
        json_obj (dict): The JSON object containing actions.

    Returns:
        any: The extracted actions or an empty list if none exist.
    """
    if json_obj is None: return []
    return json_obj['actions'] if json_obj else []



def get_missing_actions(a1: list, a2: list) -> Optional[list]:
    """
    Identifies missing actions by comparing two lists of actions.

    Args:
        a1 (list): The first list of actions.
        a2 (list): The second list of actions.

    Returns:
        Optional[list]: A list of missing actions, or None if no actions are missing.
    """
    if len(a1) >= len(a2): return None
    # Convert dictionaries to frozen-sets of items for comparison
    set_a1 = {frozenset(d.items()) for d in a1}
    set_a2 = {frozenset(d.items()) for d in a2}
    # Get the difference and convert back to dictionaries
    diff_set = set_a2 - set_a1
    return [dict(fs) for fs in diff_set]