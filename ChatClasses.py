import requests, re, socket, json, time
from errors import *
from json.decoder import JSONDecodeError

class Chat:
    def __init__(self, chat, title, duration):
        self.chat = chat
        self.title = title
        self.duration = duration

        self._output_writer = None
        self._output_callback = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            msgs = next(self.chat)
            return msgs
        except StopIteration:
            raise StopIteration

class ChatGenerator:
    _valid_url_regex = '(?x) https?://' \
                       '(?:(?:(?:www|go|m)\.)' \
                       '?twitch\.tv/videos/)' \
                       '(?P<id>\d{10})'

    _session_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
        'Accept-Language': 'en-US, en, *'  # 'de-CH'#'fr-CH'#
    }
    twitch_gql_client_id = 'kimne78kx3ncx6brgo4mv6wki5h1ko'
    _gql_api_template = f'https://api.twitch.tv/v5/videos/{{}}/comments?client_id={twitch_gql_client_id}'
    _gql_api_url = 'https://gql.twitch.tv/gql'
    _gql_api_headers = {'Content-Type': 'text/plain;charset=UTF-8',
                        'Client-ID': twitch_gql_client_id}

    _operation_hashes = {'VideoMetadata': '226edb3e692509f727fd56821f5653c05740242c82b0388883e0c0e75dcbf687'}

    _session_proxies = {}
    _max_connection_attempts = 10
    _connection_attempt_delay = .5
    _comment_dict = {'time': None,
                     'comment': None,
                     'commenter': None}

    def __init__(self):
        self.session = requests.session()
        pass

    def is_valid_url(self, url):
        match = None
        if isinstance(url, str):
            match = re.search(self._valid_url_regex, url)
        return match

    def return_gql_operation(self, video_id):
        gql_op = {'operationName': 'VideoMetadata',
                  'variables': {
                      'channelLogin': '',
                      'videoID': f'{video_id}'}}
        gql_op['extensions'] = {
            'persistedQuery': {
                'version': 1,
                'sha256Hash': self._operation_hashes[gql_op['operationName']]}}
        return self.session.post(self._gql_api_url, json=gql_op, headers=self._gql_api_headers).json()

    def _return_data_dict(self, comment_dict) -> dict:
        return {'time': comment_dict['content_offset_seconds'],
                'msg': comment_dict['message']['body'].lower(),
                'commenter': comment_dict['commenter']['display_name'],
                'commenter_id': comment_dict['commenter']['_id']}

    def _return_chat(self, vod_id:str, end_time:int):
        cursor, content_offset = '', 0

        while 1:
            api_url = f'{self._gql_api_template.format(vod_id)}&cursor={cursor}&content_offset_seconds={content_offset}'
            comments = None
            for attempt in range(self._max_connection_attempts):
                try:
                    comments = self.session.get(api_url).json()
                    break
                except JSONDecodeError:
                    print(f'Connection attempt #{attempt} failed, attempting to connect again')
                    time.sleep(self._connection_attempt_delay)
                if not comments:
                    raise CommentsNotFound('No Comments Found')

            for comment in comments['comments']:
                #_comment_dict = {'time', 'comment, commenter, commenter_id}
                comment_relevant_info = self._return_data_dict(comment)
                comment_time = comment_relevant_info.get('time')
                if comment_time is not None:
                    if comment_time < 0:
                        continue
                    elif comment_time < 0:
                        return
                yield comment_relevant_info

            cursor = comments.get('_next')
            if not cursor:
                return

    def return_chat(self, url) -> Chat:
        url_match = self.is_valid_url(url)
        if url_match is None:
            raise InvalidURL('Invalid URL, check that its correct.')
        vod_id = url_match.string[url_match.regs[1][0]:url_match.regs[1][1]]
        video = self.return_gql_operation(vod_id)['data']['video']
        if not video:
            raise VideoUnavailable(
                "Sorry. Unless you've got a time machine, that content is unavailable.")
        duration, title = video.get('lengthSeconds'), video.get('title')
        return Chat(self._return_chat(vod_id, duration), title, duration)





if __name__ == '__main__':
    vod_url = 'https://www.twitch.tv/videos/1611458439'
    chat = ChatGenerator().return_chat(vod_url)
    for msg in chat:
        print(msg['msg'])

    print('b')
