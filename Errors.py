class Error(Exception):
    pass

class InvalidURL(Error):
    pass

class VideoUnavailable(Error):
    pass

class CommentsNotFound(Error):
    pass

class AudioDownloadError(Error):
    pass