"""Modified from https://github.com/qiuqiangkong/mini_llm/blob/main/sample.py
"""
from __future__ import annotations

import librosa


from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
import torch
import torch.nn as nn
from panns_inference import AudioTagging
from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from models.llama import Llama, LlamaConfig


def inference(audio_path):
    ckpt_path = "inference\\step=20000.pth"
    sr = 32000
    device = "cuda"
    max_length = 20
    audio_encoder_name = "Cnn14"
    llm_decoder_name = "Llama"
    num_samples = 1
    temperature = 1.0
    top_k = 200

    # crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)
    target_transform = [
        TextNormalization(),
        BertTokenizer(max_length=max_length)
    ]
    tokenizer = target_transform[1].tokenizer
    start_token_id = tokenizer.cls_token_id
    text_vocab_size = tokenizer.vocab_size

    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    audio_encoder.to(device)
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name,
        audio_latent_dim=audio_latent_dim,
        text_vocab_size=text_vocab_size
    )
    llm_decoder.load_state_dict(torch.load(ckpt_path))
    llm_decoder.to(device)
    text_ids = torch.LongTensor([[start_token_id]]).to(device)

    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    audio = torch.Tensor(audio[None, None, :]).to(device)
    audio_latent = get_audio_latent(
        model_name=audio_encoder_name,
        model=audio_encoder, audio=audio
    )
    for n in range(num_samples):
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]
        with torch.no_grad():
            llm_decoder.eval()
            outputs = llm_decoder.generate(
                seqs=input_seqs,
                seq_types=seq_types,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k
            )
        sampled_text_ids = outputs[-1][0].cpu().numpy()
        strings = tokenizer.decode(token_ids=sampled_text_ids, skip_special_tokens=True)
        return strings


def tokens_to_string(tokens, tokenizer):
    return "".join([tokenizer.itos(token) for token in tokens])

def get_audio_encoder(model_name: str) -> nn.Module:
    r"""Load pretrained audio encoder."""
    if model_name == "Cnn14":
        model = AudioTagging().model
        latent_dim = 2048
        return model, latent_dim

    else:
        raise ValueError(model_name)


def get_llm_decoder(
    model_name: str, 
    audio_latent_dim: int, 
    text_vocab_size: int
) -> nn.Module:
    r"""Initialize LLM decoder."""
    if model_name == "Llama":
        config = LlamaConfig(
            block_size=1024,
            audio_latent_dim=audio_latent_dim,
            vocab_size=text_vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return Llama(config=config)

    else:
        raise ValueError(model_name)    


def get_audio_latent(
    model_name: str, 
    model: nn.Module, 
    audio: torch.Tensor
) -> torch.Tensor:
    r"""Calculate audio latent from an audio.

    Args:
        model_name: str
        model: nn.Module
        audio: (batch_size, channels_num, samples_num)

    Outputs:
        audio_latent: (batch_size, time_steps, emb_dim)
    """
    if model_name == "Cnn14":
        with torch.no_grad():
            model.eval()
            latent = model(audio[:, 0, :])["embedding"]  # (b, d)
            latent = latent[:, None, :]  # (b, t_audio, d)
        return latent

    else:
        raise ValueError(model_name)        


if __name__ == "__main__":

    audio_path = "recoding"

    inference(audio_path)