import os.path
path = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

params_base_dir = f'{path}/content_based_params'
common_params = f'{params_base_dir}/common'
game_params = f'{params_base_dir}/games'
streamer_params = f'{params_base_dir}/streamers'
temp_audio = f'temp.mkv'
clips_dir = f'{path}/clips'