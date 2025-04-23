import sys

# Main prompt template function
prompt_follow_up = lambda email_thread, datetime, options, username: \
f'''
You are an AI-driven Email Analyzer and Generator specializing in the examination of user's EMAIL THREADS (both sent and received messages), generating responses that are consistent and relevant to the content of the EMAIL THREAD.

## TASK:
<task>
Given a list of possible RESPONSE TYPES to generate:
1. Analyze the EMAIL THREAD conversation.
2. Choose EXACTLY AND ONLY ONE RESPONSE TYPE.
3. Generate an email response that is appropriate and consistent with:
- The CURRENT WEEK DAYS.
- The EMAIL THREAD "content" field of its "messages".
- The RESPONSE TYPE to be generated.
IMPORTANT: The generated response MUST BE the reply to the last message in the EMAIL THREAD.
</task>

## CURRENT WEEK DAYS:
<current_week_days>
{datetime.get('current_week_days')}
</current_week_days>


## EMAIL THREAD:
<email_thread>
{format_email_thread(email_thread)}
</email_thread>


## GENERAL INSTRUCTIONS and RULES:
<instructions>
1. Analyze the CURRENT DATE and TIME of this exact moment. To do this, examine the <current_datetime> tag. Also, analyze the days of the current week.
2. Analyze the content of EMAIL THREAD messages. Pay close attention to the most recent message ("order_index" field).
3. Decide which RESPONSE TYPE best fits the most recent message in the EMAIL THREAD. Choose EXACTLY AND ONLY ONE of the following RESPONSE TYPES.
RESPONSE TYPES:

{format_options(options)}

4. Based on the RESPONSE TYPE CHOSEN, generate a response that directly addresses the most recent message in the EMAIL THREAD. Ensure that the response is relevant to the message content and follows the corresponding RESPONSE TYPE-specific rules.
5. The generated response MUST follow the rules contained in the <rules> tag of the RESPONSE TYPE CHOSEN.
6. The response generated MUST BE DISCURSIVE, have a FORMAL TONE, and BE CONCISE.
7. When in the messages the "sender" is "Me" it means that the "sender" is "{username}".
8. The generated response MUST refer to the most recent message in the EMAIL THREAD and MUST resemble the REPLY WRITTEN BY "{get_sender(email_thread)}".
9. The generated response MUST be the reply written by "{get_sender(email_thread)}".
10. The generated response MUST be the reply that the human "{username}" will receive.
</instructions>

<rules>
** IMPORTANT: **
1. The following constraints MUST BE met:
- Choose ONLY ONE RESPONSE TYPE.
- The generated response MUST STRICTLY FOLLOW the rules and purpose of the CHOSEN RESPONSE TYPE.
- The generated response MUST ONLY REFER to the CHOSEN RESPONSE TYPE.
- ** AVOID to mention or include elements from other RESPONSE TYPES. **
</rules>

{format_rules(options, datetime)}

## OUTPUT FORMAT RULES:
<output_format_rules>
1. Return ONLY a JSON object with the generated response.
2. The answer MUST have exactly the following structure:

{{
   "id": string,  // ID of the response TYPE
   "response": string  // Generated response
   "event_original_date": string  // Original event date (JUST FOR 'EVENT_RESCHEDULED', null for other TYPES)
   "event_new_date": string  // New event date (JUST FOR 'EVENT_RESCHEDULED')
   "day_difference": string  // event_new_date - event_original_date (JUST FOR 'EVENT_RESCHEDULED', null for other TYPES)
   "time_difference": string // event new hour - event original hour (JUST FOR 'EVENT_RESCHEDULED', null for other TYPES)
}}

3. Return ONLY the JSON object, no explanations. 
4. MUST maintain the SAME JSON structure.
</output_format_rules>

Think step by step before generating the answer.
First, think about which RESPONSE TYPES is best suited to the EMAIL THREAD and in particular the last message, and CHOOSE EXACTLY AND ONLY ONE RESPONSE TYPE.
Then think about the best response to generate to reply to the last message in the EMAIL THREAD. 
Use the <task>, <current_week_days>, <email_thread>, <instructions> and <rules> tags to best understand and process the best and correct response.
The response MUST mandatory follow the exactly JSON structure described in the <output_format_rules> tag.
The following constraints MUST BE met:
- The generated response MUST simulate the REPLY WRITTEN BY "{get_sender(email_thread)}".
- The generated response MUST simulate the REPLY THAT "{username}" WILL RECEIVE.
 

Please, respond ONLY and JUST with the single JSON object. AVOID to add extra text.

JSON Answer:
'''



#
# Lambda functions for different response types
#
EVENT_RESCHEDULED = lambda ctx:\
f'''
## RULES FOR 'EVENT_RESCHEDULED' TYPE:
<rules>
1. Knowing that NOW is {ctx.get('current_date')} and it is {ctx.get('current_time')}, FIND the DATE and TIME of the EVENT. To do this, you MUST consider the "content" the EMAIL THREAD.  
2. IMPORTANT: The EVENT'S DATE and TIME is WITHIN the message content of the EMAIL THREAD. 
3. STRONGLY AVOID TO CONSIDER the messages "timestamp".
4. The generated response MUST mandatory have a DATETIME DIFFERENT from that contained in the EMAIL THREAD message. 
5. The response MUST reschedule the event to a DIFFERENT DATE and DIFFERENT TIME than the original one.
6. The response MUST reschedule the event to a RANDOM DATE BETWEEN THOSE FOUND IN “current_week_days”.
7. The response MUST CONTAIN the new date.
8. The response MUST CONTAIN the new datetime, namely the day of the week, the calendar date, and the time.
</rules>
'''


EVENT_ACCEPTED = lambda ctx:\
f'''
## RULES FOR 'EVENT_ACCEPTED' TYPE:
<rules>
1. FIND the NEW DATETIME of the EVENT. To do this, you MUST consider the most recent EMAIL THREAD message "content".
2. The response MUST MUST ACCEPT the specific proposed new datetime for the EVENT. 
3. The response MUST CONTAIN the new datetime, namely the day of the week, the calendar date, and the time.
4. The response MUST CONTAIN exactly the SAME datetime mentioned in the most recent message in the EMAIL THREAD.
5. STRONGLY AVOID TO GENERATE others datetime.
</rules>
'''


EVENT_CANCELLED = lambda ctx:\
f'''
## RULES FOR 'EVENT_CANCELLED' TYPE:
<rules>
1. The response MUST communicate and emphasize the CANCELLATION of the event.
2. STRONGLY AVOID TO RESCHEDULE the event. The event MUST BE CANCELLED.
</rules>
'''


TASK_ADDED = lambda ctx:\
f'''
## RULES FOR 'TASK_ADDED' TYPE:
<rules>
1. The response MUST contain an ADDITIONAL TASK alongside the existing ones.
2. Examples of additional tasks:
- Creation of a relevant document.
- A presentation.
- etc.
</rules>
'''


GENERAL_ACKNOWLEDGMENT = lambda ctx:\
f'''
## RULES FOR 'GENERAL_ACKNOWLEDGMENT' TYPE:
<rules>
1. The response MUST describe a simple an short acknowledgment message. For example, the response might include:
- An acknowledgement message (if it only serves to confirm receipt)
- An appreciation message (if it expresses positive feedback)
- A courtesy message (if it only serves to close the communication in a polite way.
** IMPORTANT: The response MUST BE JUST AN ACKNOWLEDGMENT.
</rules>
'''



def format_datetime(datetime):
    """
    Formats a datetime dictionary into a string representation.

    Args:
        datetime (dict): Dictionary containing datetime information.

    Returns:
        str: A string with each key-value pair from the datetime dictionary formatted as "key":"value".
    """
    return ''.join([f'"{k}":"{datetime[k]}"\n' for k in datetime])



def format_email_thread(email_thread):
    """
    Formats an email thread into a string representation.

    Args:
        email_thread (dict): Dictionary containing email thread data with 'id' and 'messages' keys.
            The 'messages' key should contain a list of message dictionaries.

    Returns:
        str: A string representation of the email thread with each key-value pair on a new line.
            Messages are formatted as a series of nested key-value pairs.
    """
    e = dict(email_thread)
    e['id'] = None
    e['messages'] = "".join(['\n{\n' + ''.join([f'"{k}":"{msg[k]}"\n' for k in msg]) + '},' for msg in e['messages']]) or e.pop('messages', None)
    return ''.join([f'"{k}":"{e[k]}"\n' for k in e if e[k] is not None])



def format_options(options):
    """
    Formats a list of options into a bullet-point string.

    Args:
        options (list): List of option items to format.

    Returns:
        str: A string with each option prefixed by '- ' on a new line.
    """
    return "\n".join([f'- {o}' for o in options])



def format_rules(options, datetime):
    """
    Formats rules for each option type using the corresponding function.

    Args:
        options (list): List of option dictionaries, each with an 'id' key.
        datetime (dict): Dictionary containing datetime context information.

    Returns:
        str: A string with all formatted rules joined with double newlines.
    """
    return "\n\n".join([getattr(sys.modules[__name__], f'{option["id"]}')(ctx=datetime) for option in options])



def get_sender(email_thread):
    """
    Gets the sender of the first message in an email thread.

    Args:
        email_thread (dict): Dictionary containing email thread data with a 'messages' key.
            The 'messages' key should contain a list of message dictionaries, each with a 'sender' key.

    Returns:
        str: The sender of the first message in the email thread.
    """
    return email_thread['messages'][0]['sender']