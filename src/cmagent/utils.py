from typing import Any, Dict, Optional, Tuple
import re
import logging
import dirtyjson


def extract_json_codeblock(md_text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    match = re.search(r"```json[^\n]*\r?\n(.*?)\r?\n?```", md_text, re.DOTALL | re.IGNORECASE)
    if not match:
        msg = "❌ extract_json_codeblock: can't find json block"
        logging.error(msg)
        return {}, msg

    block = match.group(1).strip()
    try:
        result, error = safe_json_parse(block)
    except Exception as e:
        msg = f"❌ extract_json_codeblock: safe_json_parse crashed: {e}"
        logging.exception(msg)
        return {}, msg

    if isinstance(result, dict):
        return result, None

    msg = "❌ extract_json_codeblock: failed to parse JSON block"
    detail = error if error else f"Parsed result isn't dict (got {type(result).__name__})"
    logging.error("%s\n↳ Error detail: %s\n↳ (block excerpt, first 200 chars): %r",
                  msg, detail, block[:200])
    return {}, detail


def safe_json_parse(json_text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Attempts to parse a JSON string. Automatically fixes common errors and returns a dictionary of results.
    On failure, returns None and a detailed error message (which will not include the full original text).
    """
    try:
        parsed_obj = dirtyjson.loads(json_text)
        return parsed_obj, None
    except Exception as e:
        error_message = f"❌ Failed to parse JSON even with dirtyjson: {str(e)}"
        return None, error_message
