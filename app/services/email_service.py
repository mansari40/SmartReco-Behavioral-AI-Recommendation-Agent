from email.message import EmailMessage
import smtplib

from app.config import settings
from app.security import create_password_reset_token


def send_password_reset_email(user) -> None:
    if not settings.smtp_host or not settings.smtp_user or not settings.smtp_password:
        return

    token = create_password_reset_token(user.id)
    reset_url = f"{settings.public_base_url.rstrip('/')}/password-reset?token={token}"
    body = f"Hi {user.first_name or user.email},\n\n" \
           "We received a request to reset your SmartReco password. " \
           "Click the link below to choose a new password:\n\n" \
           f"{reset_url}\n\n" \
           "If you didn't request a password reset, you can ignore this message.\n\n" \
           "Thanks,\nSmartReco Team\n"

    msg = EmailMessage()
    msg['Subject'] = 'Reset your SmartReco password'
    msg['From'] = settings.smtp_from
    msg['To'] = user.email
    msg.set_content(body)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
        if settings.smtp_use_tls:
            smtp.starttls()
        smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)