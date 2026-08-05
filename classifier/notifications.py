import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

_CONTROLLER_BASE = "https://controller.gnometrading.group/predictions/events"


def _event_link(event_id: int, name: str) -> str:
    return f"<{_CONTROLLER_BASE}/{event_id}|{name}>"


def _event_section(label: str, events: list[tuple[int, str]]) -> dict:
    lines = "\n".join(f"• {_event_link(eid, name)}" for eid, name in events[:20])
    if len(events) > 20:
        lines += f"\n_... and {len(events) - 20} more_"
    return {"type": "section", "text": {"type": "mrkdwn", "text": f"*{label}*\n{lines}"}}


def format_notification_blocks(
    created_events: list[tuple[int, str]],
    resolved_events: list[tuple[int, str]],
) -> list[dict]:
    blocks: list = [
        {"type": "header", "text": {"type": "plain_text", "text": "Contract Classifier"}},
    ]

    if created_events:
        n = len(created_events)
        blocks.append(_event_section(f"{n} new event{'s' if n != 1 else ''}", created_events))
    if resolved_events:
        n = len(resolved_events)
        blocks.append(_event_section(f"{n} event{'s' if n != 1 else ''} resolved", resolved_events))

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
