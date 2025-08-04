def send_console_alert(message):
    print(f"[ALERT] {message}")

def send_email_alert(recipient_email, subject, body):
    # Placeholder: tutaj możesz dodać prawdziwą wysyłkę przez SMTP
    print(f"[EMAIL to {recipient_email}] {subject}: {body}")
