from sandbox.notifications.webhook import WebhookNotifier

def get_notifier() -> WebhookNotifier:
    """Returns a configured webhook notifier instance."""
    return WebhookNotifier()
