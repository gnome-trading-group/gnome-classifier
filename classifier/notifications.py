import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


def format_notification_blocks(
    new_symbols: list[str],
    entity_counts: dict,
    resolution_counts: dict,
    relationships_written: int,
) -> list[dict]:
    blocks: list = [
        {"type": "header", "text": {"type": "plain_text", "text": "Contract Classifier"}},
    ]

    if new_symbols:
        lines = "\n".join(f"• `{s}`" for s in new_symbols)
        count = len(new_symbols)
        noun = "security" if count == 1 else "securities"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{count} new {noun}*\n{lines}"},
        })

    parts = []
    if entity_counts.get("events_created"):
        parts.append(f"{entity_counts['events_created']} events")
    if entity_counts.get("securities_created"):
        parts.append(f"{entity_counts['securities_created']} securities")
    if entity_counts.get("listings_created"):
        parts.append(f"{entity_counts['listings_created']} listings")
    if relationships_written:
        parts.append(f"{relationships_written} relationships")
    if resolution_counts.get("events_resolved"):
        parts.append(f"{resolution_counts['events_resolved']} events resolved")
    if resolution_counts.get("securities_deactivated"):
        parts.append(f"{resolution_counts['securities_deactivated']} securities deactivated")
    if parts:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": " · ".join(parts)}],
        })

    return blocks


def send_slack_notification(token: str, channel: str, blocks: list[dict]) -> bool:
    payload = json.dumps({"channel": channel, "blocks": blocks}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    if not body.get("ok"):
        logger.error("Slack notification failed: %s", body.get("error"))
        return False
    logger.info("Slack notification sent")
    return True
