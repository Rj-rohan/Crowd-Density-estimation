import os
import time
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN")
FROM_WHATSAPP = os.getenv("TWILIO_FROM", "whatsapp:+14155238886")
TO_WHATSAPP   = os.getenv("TWILIO_TO")
CONTENT_SID   = os.getenv("TWILIO_CONTENT_SID")

COOLDOWN_SECONDS = 60   # send at most 1 WhatsApp alert per minute

_last_sent: float = 0


def send_whatsapp_alert(count: int, category: str) -> bool:
    """
    Send a WhatsApp alert via Twilio.
    Returns True if sent, False if still in cooldown or misconfigured.
    """
    global _last_sent

    if not all([ACCOUNT_SID, AUTH_TOKEN, TO_WHATSAPP, CONTENT_SID]):
        print("[Alert] Twilio credentials missing in .env — skipping")
        return False

    now = time.time()
    if now - _last_sent < COOLDOWN_SECONDS:
        return False

    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        client.messages.create(
            from_=FROM_WHATSAPP,
            to=TO_WHATSAPP,
            content_sid=CONTENT_SID,
            content_variables=f'{{"1":"{count}","2":"{category}"}}',
        )
        _last_sent = now
        print(f"[Alert] WhatsApp sent — {count} people, {category} density")
        return True
    except Exception as e:
        print(f"[Alert] Twilio error: {e}")
        return False
