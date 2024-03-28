import logging


class ExcludeMessagesFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        return "Found credentials" not in message and "Object not found" not in message


def configure_logger():
    # Create logger
    logger = logging.getLogger(__name__)

    # Create filter instance
    exclude_filter = ExcludeMessagesFilter()

    # Set logging level and format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("log_file.log"),
            logging.StreamHandler(),
        ],
    )

    # Add filter to the handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(exclude_filter)

    return logger
