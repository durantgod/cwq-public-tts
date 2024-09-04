import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def main():
    output_path = os.path.dirname(os.path.abspath(__file__))
    dataset_config = BaseDatasetConfig(
        formatter="ipa_format",
        path="D://github//TTS//recipes//ljspeech//glow_tts",
        meta_file_train="D://github//TTS//recipes//ljspeech//glow_tts//metadata.csv",
    )

    config = GlowTTSConfig(
        # Configuration parameters
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        # Loading samples parameters
    )
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    trainer.fit()

if __name__ == '__main__':
    main()