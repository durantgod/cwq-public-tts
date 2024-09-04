import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="哈哈，每次读到这首诗，就感慨良深。很多时候，由于各种原因，我们写文章，并不不能想什么就说什么，直抒胸臆。尤其是在曹雪芹那个年代，文字狱很可怕，不得不经常用隐喻的手法，草蛇灰线，伏笔千里。表面看起来荒唐，但在字里行间，总会透露出作者的真情实感，只要用心去读，认真去体会，就能感悟到作者的一片苦心。", speaker_wav="durant.wav", language="zh-cn")
# Text to speech to a file
tts.tts_to_file(text="你发如雪，凄美了离别", speaker_wav="durant.wav", language="zh-cn", file_path="output2.wav")