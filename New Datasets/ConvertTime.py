import pytz
from datetime import datetime
# Define a mapping from time zone abbreviations to pytz time zones
timezone_abbreviations = {
    'UTC': 'UTC',
    'GMT': 'GMT',
    'IST': 'Asia/Kolkata',       # Indian Standard Time
    'PCT': 'Asia/Shanghai',      # Assuming PCT refers to China Standard Time (Beijing Time)
    # Add more mappings as needed
}
def convert_timezone(dt_str, from_tz_abbr, to_tz_abbr_list):
    """
    Convert datetime string from one timezone to others.

    :param dt_str: The datetime string in YYYY-MM-DD HH:MM:SS format.
    :param from_tz_abbr: Abbreviation of the source timezone.
    :param to_tz_abbr_list: List of abbreviations of target timezones.
    :return: Dictionary with target timezone abbreviations as keys and converted datetime strings as values.
    """
    # Parse the input datetime string
    naive_dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    # Get the source timezone object
    from_tz_name = timezone_abbreviations.get(from_tz_abbr.upper())
    if not from_tz_name:
        raise ValueError(f"Unknown source timezone abbreviation: {from_tz_abbr}")
    from_tz = pytz.timezone(from_tz_name)
    # Localize the naive datetime to the source timezone
    localized_dt = from_tz.localize(naive_dt)
    converted_times = {}
    for to_tz_abbr in to_tz_abbr_list:
        to_tz_name = timezone_abbreviations.get(to_tz_abbr.upper())
        if not to_tz_name:
            raise ValueError(f"Unknown target timezone abbreviation: {to_tz_abbr}")
        to_tz = pytz.timezone(to_tz_name)
        # Convert the datetime to the target timezone
        converted_dt = localized_dt.astimezone(to_tz)
        converted_times[to_tz_abbr.upper()] = converted_dt.strftime('%Y-%m-%d %H:%M:%S')
    return converted_times
