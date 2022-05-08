class NoTrainingInProgressException(Exception):
    def __init__(self, message):
        super(NoTrainingInProgressException, self).__init__(message)