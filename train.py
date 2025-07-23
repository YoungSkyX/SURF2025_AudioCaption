import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from audidata.datasets import Clotho
from audidata.io.crops import RandomCrop
from audidata.samplers import InfiniteSampler, PseudoRandomSampler
from audidata.transforms import Mono
from panns_inference import AudioTagging
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from models.llama import Llama, LlamaConfig


def train(args):
    
    # Default parameters
    sr = 32000  # To be consistent with the encoder
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-4
    test_every_n_steps = 200
    save_every_n_steps = 2000
    training_steps = 20000
    wandb_log = True
    device = "cuda"
    max_length = 30  # Max caption length
    clip_duration = 10.  # Audio clip duration
    audio_encoder_name = "Cnn14"
    llm_decoder_name = "Llama"

    filename = Path(__file__).stem
    
    # Dataset
    root = "/root/autodl-tmp/datasets/clotho" # Change to local path

    # Checkpoints directory
    model_name = "{}_{}".format(audio_encoder_name, llm_decoder_name)
    ckpts_dir = Path("./checkpoints", filename, model_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Audio Cropper
    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)

    # Caption transforms
    target_transform = [
        TextNormalization(),  # Remove punctuations
        BertTokenizer(max_length=max_length)  # Convert captions to token IDs
    ]
    pad_token_id = target_transform[1].tokenizer.pad_token_id  # 0
    text_vocab_size = target_transform[1].tokenizer.vocab_size  # 30,522

    # Datasets
    train_dataset = Clotho(
        root=root,
        split="train",
        sr=sr,
        crop=crop,
        transform=Mono(),
        target_transform=target_transform
    )

    test_dataset = Clotho(
        root=root,
        split="test",
        sr=sr,
        crop=crop,
        transform=Mono(),
        target_transform=target_transform
    )

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)
    eval_train_sampler = PseudoRandomSampler(train_dataset)
    eval_test_sampler = PseudoRandomSampler(test_dataset)
    
    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    eval_train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=eval_train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    eval_test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=eval_test_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Pretrained audio encoder
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    audio_encoder.to(device)

    # LLM decoder
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name, 
        audio_latent_dim=audio_latent_dim, 
        text_vocab_size=text_vocab_size
    )
    llm_decoder.to(device)

    # Optimizer
    optimizer = optim.AdamW(params=llm_decoder.parameters(), lr=learning_rate)

    if wandb_log:
        wandb.init(project="mini_audio_caption", mode="offline", name="{}".format(model_name))
        # Set to offline mode

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        audio = data["audio"].to(device)  # shape: (b, c, t_audio)
        text_ids = data["target"].to(device)  # shape: (b, t_text)
        
        # Extract audio embeddings
        audio_latent = get_audio_latent(
            model_name=audio_encoder_name, 
            model=audio_encoder, audio=audio
        )
        
        # Combine audio embeddings and text ids
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]

        # Forward
        llm_decoder.train()

        output_seqs = llm_decoder(
            seqs=input_seqs,
            seq_types=seq_types,
            mask=None
        )
        # list of output, e.g., [(b, t_audio, audio_dim), (b, t_text, vocab_size)]

        # Loss
        loss = caption_loss(
            output_seqs=output_seqs, 
            input_seqs=input_seqs, 
            ignore_index=pad_token_id
        )

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        loss.backward()     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        # Evaluate
        if step % test_every_n_steps == 0:

            train_loss = validate(
                dataloader=eval_train_dataloader, 
                audio_encoder_name=audio_encoder_name, 
                audio_encoder=audio_encoder, 
                llm_decoder=llm_decoder, 
                pad_token_id=pad_token_id
            )
            test_loss = validate(
                dataloader=eval_test_dataloader, 
                audio_encoder_name=audio_encoder_name, 
                audio_encoder=audio_encoder, 
                llm_decoder=llm_decoder, 
                pad_token_id=pad_token_id
            )

            print("------ step: {} ------".format(step))
            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )
        
        # Save model
        if step % save_every_n_steps == 0:
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(llm_decoder.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == training_steps:
            break
        

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


def caption_loss(
    output_seqs: list[torch.Tensor], 
    input_seqs: list[torch.Tensor], 
    ignore_index: int
) -> torch.float:
    r"""Calculate caption loss."""

    output_audio_logits, output_text_logtis = output_seqs
    target_audio_latents, target_text_ids = input_seqs

    loss = F.cross_entropy(
        input=output_text_logtis[:, 0 : -1, :].flatten(0, 1),  # shape: (B*T, V)
        target=target_text_ids[:, 1 :].flatten(0, 1),  # shape: (B*T,)
        ignore_index=ignore_index
    )

    return loss


def validate(
    dataloader: DataLoader, 
    audio_encoder_name: str,
    audio_encoder: nn.Module, 
    llm_decoder: nn.Module, 
    pad_token_id: int,
    valid_steps=10
) -> float:
    r"""Validate the model on part of data."""

    device = next(audio_encoder.parameters()).device
    losses = []

    for step, data in enumerate(dataloader):

        # Move data to device
        audio = data["audio"].to(device)  # shape: (b, t_audio)
        text_ids = data["target"].to(device)  # shape: (b, t_text)
        
        # Extract audio embeddings
        audio_latent = get_audio_latent(model_name=audio_encoder_name, model=audio_encoder, audio=audio)
        
        # Combine audio embeddings and text ids
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]

        # Forward
        llm_decoder.eval()
        outputs = llm_decoder(
            seqs=input_seqs,
            seq_types=seq_types,
            mask=None
        )

        # Loss
        loss = caption_loss(
            output_seqs=outputs, 
            input_seqs=input_seqs, 
            ignore_index=pad_token_id
        )
        losses.append(loss.item())

        if step == valid_steps:
            break

    return np.mean(losses)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train(args)