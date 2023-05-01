class PreTrainingException(Exception):
    """
    Custom exception in our module. Should be used when an incoerent
    flow is identified.
    """

    def __init__(self, message, *args: object) -> None:
        self.message = message
        super().__init__(message, *args)


class UnexistingSourceException(Exception):
    """
    Custom exception in our module. Should be used when the source models
    needed for training don't yet exist.
    """

    def __init__(self, *args: object) -> None:
        self.message = 'Please train the needed source models before trying to transfer.'
        super().__init__(self.message, *args)


class NotLoadedException(PreTrainingException):
    """
    Custom exception in our module. Should be used when the flow for
    loading data is not respected.
    """

    def __init__(self, *args: object) -> None:
        self.message = 'The data is not loaded. Please check your code.'
        super().__init__(self.message, *args)


class CatalogNotFoundException(PreTrainingException):
    """
    Custom exception in our module. Should be used when there is no
    catalog in the models folder.
    """

    def __init__(self, *args: object) -> None:
        self.message = 'The action cannot be executed until the catalog is built.'
        super().__init__(self.message, *args)


class UndefinedOutputException(PreTrainingException):
    """
    Custom exception in our module. Should be used when trying to save
    a model but the directory to save it has not been specified.
    """

    def __init__(self, *args: object) -> None:
        self.message = 'The model cannot be saved without a path.'
        super().__init__(self.message, *args)


class WrongParamsMappingException(PreTrainingException):
    """
    Custom exception in our module. Should be used to raise errors while
    mapping items from one dataset to another.
    """

    def __init__(self, *args: object) -> None:
        self.message = 'It is not possible to execute this action with the given parameters.'
        super().__init__(self.message, *args)
