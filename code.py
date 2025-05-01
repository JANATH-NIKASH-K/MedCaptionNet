import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import numpy as np
import math
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import torch.nn.functional as F
import random  # For scheduled sampling
import torch.optim as optim  # For optimizer
from rouge import Rouge  # For ROUGE evaluation
from torchvision.transforms import ToPILImage

# --- Debugging Captions and Tokenizer ---
def debug_tokenizer(tokenizer):
    print("Tokenizer Special Tokens:")
    print("PAD:", tokenizer.pad_token_id)
    print("CLS:", tokenizer.cls_token_id)
    print("SEP:", tokenizer.sep_token_id)

def debug_generated_captions(generated_captions):
    print("Generated Captions:")
    for caption in generated_captions:
        print(caption)

def visualize_attention(image, output_dir, image_id, attention_weights, processed_ids):
    """
    Visualizes attention map over the input image.

    Args:
        image: The input image (Tensor or PIL image).
        attention_weights: Tensor of shape [1, 1, num_patches].
        processed_ids: Set of already processed image IDs.
    """
    save_path = os.path.join(output_dir, f"attention_map_{image_id}.png")
    
    # Check if the heatmap already exists or the image ID is processed
    if os.path.exists(save_path) or image_id in processed_ids:
        print(f"Skipping attention heatmap for image ID {image_id}. Already processed.")
        return

    num_patches = attention_weights.shape[-1]
    feature_map_size = int(math.sqrt(num_patches))

    if feature_map_size ** 2 != num_patches:
        raise ValueError(f"Attention map not square: got {num_patches} elements.")

    attention_map = attention_weights.view(1, 1, feature_map_size, feature_map_size)

    attention_map_resized = torch.nn.functional.interpolate(
        attention_map, size=(image.shape[-2], image.shape[-1]), mode="bilinear", align_corners=False
    ).squeeze().cpu().detach().numpy()

    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / \
                            (attention_map_resized.max() - attention_map_resized.min() + 1e-8)

    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[0] == 1:
            image = image.squeeze(0).cpu().numpy()
        else:
            image = image.permute(1, 2, 0).cpu().numpy()

    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_GRAY2BGR), 0.7, heatmap, 0.3, 0)

    cv2.imwrite(save_path, overlayed_img)
    print(f"Attention heatmap saved at: {save_path}")

    # Add the image ID to the processed set
    processed_ids.add(image_id)

def train_model(
    model, dataloader, criterion, optimizer, device, num_epochs=10
):
    model.to(device)
    model.train()

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0  # Track loss for the current epoch

        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            captions = batch['input_ids'].to(device)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            logits, _ = model(images, captions)  # Unpack logits
            targets = captions[:, 1:]  # Shift targets for teacher forcing
            logits = logits[:, :targets.size(1), :]  # Align logits with target length

            # Compute the loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),  # Flatten logits for CrossEntropyLoss
                targets.contiguous().view(-1)  # Flatten targets
            )

            # Backward pass and optimization
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model parameters
            optimizer.step()

            # Accumulate loss for the epoch
            epoch_loss += loss.item()

        # Calculate average epoch loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Adjust learning rate based on average loss
        lr_scheduler.step(avg_loss)

    print("Training complete!")

def generate_caption(
    model, 
    tokenizer, 
    image, 
    device, 
    output_dir, 
    image_id, 
    max_seq_len=30, 
    beam_width=5, 
    temperature=1.0, 
    repetition_penalty=1.2,
    processed_ids=None
):
    """
    Generates a caption for an image using the given model and tokenizer.

    Args:
        model: The captioning model.
        tokenizer: Tokenizer for encoding and decoding sequences.
        image: The input image (either a file path or preprocessed tensor).
        device: The device to run the model on ('cuda' or 'cpu').
        output_dir: Directory to save attention maps.
        image_id: Unique identifier for the image (for saving files).
        max_seq_len: Maximum length of the generated caption.
        beam_width: Number of beams for beam search.
        temperature: Sampling temperature (higher values produce more diverse results).
        repetition_penalty: Penalty to avoid repetitive tokens in the output.
        processed_ids: Set of already processed image IDs.

    Returns:
        caption: The generated caption as a string.
    """
    model.eval()
    with torch.no_grad():
        # Preprocess the image
        image_tensor = preprocess_image(image, device)

        # Extract features
        features = model.vision_extractor(image_tensor)
        global_features = features.mean(dim=(2, 3))
        local_features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

        # Project features to hidden dimensions
        projected_global_features = model.global_projection(global_features)
        projected_local_features = model.local_projection(local_features)

        # Prepare attention mechanism inputs
        query = projected_global_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        key = projected_local_features  # [batch, num_patches, hidden_dim]
        value = projected_local_features  # [batch, num_patches, hidden_dim]

        # Calculate attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, value).squeeze(1)

        # Initialize decoder states
        hidden_state = attended_features.unsqueeze(0).repeat(model.language_generator.num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state)
        sequences = [[tokenizer.cls_token_id]]  # Start token
        scores = torch.zeros(1, device=device)

        # Generate sequences using beam search
        for t in range(max_seq_len):
            all_candidates = []
            for i, seq in enumerate(sequences):
                input_token = torch.tensor([seq[-1]], device=device).unsqueeze(0)
                embeddings = model.embedding(input_token)
                output, (hidden_state, cell_state) = model.language_generator(
                    embeddings, (hidden_state, cell_state)
                )
                logits = model.fc(output[:, -1, :]) / temperature

                # Apply repetition penalty
                for token_id in set(seq[-3:]):  # Penalize recent tokens
                    logits[0, token_id] /= repetition_penalty

                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=min(50, probs.shape[-1]), dim=-1)  # Top-K sampling

                # Generate candidates
                for idx, prob in zip(top_indices[0], top_probs[0]):
                    candidate_seq = seq + [idx.item()]
                    candidate_score = scores[i] + torch.log(prob)
                    all_candidates.append((candidate_seq, candidate_score.item()))

            # Select top sequences based on scores
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            sequences = [x[0] for x in ordered]
            scores = torch.tensor([x[1] for x in ordered], device=device)

            # Stop if all sequences end with the SEP token
            if all(seq[-1] == tokenizer.sep_token_id for seq in sequences):
                break

        # Decode the best sequence
        best_sequence = sequences[0]
        caption = tokenizer.decode(best_sequence, skip_special_tokens=True)
        caption = caption.replace("[PAD]", "").replace("[UNK]", "").strip()

        # Visualize attention
        visualize_attention(image_tensor, output_dir, image_id, attention_weights, processed_ids)

    return caption

import os

def evaluate_model(model, tokenizer, dataset, device, transform, batch_size=8, output_dir="evaluation_outputs"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Files for writing captions
    generated_captions_file = os.path.join(output_dir, "generated_captions.txt")
    reference_captions_file = os.path.join(output_dir, "reference_captions.txt")
    metrics_file = os.path.join(output_dir, "evaluation_metrics.txt")

    # Clear previous files
    open(generated_captions_file, 'w').close()
    open(reference_captions_file, 'w').close()
    open(metrics_file, 'w').close()

    model.eval()
    references = []
    hypotheses = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    cider_scorer = Cider()
    spice_scorer = Spice()

    rouge_1_f1 = rouge_2_f1 = rouge_l_f1 = 0.0

    with torch.no_grad():
        for idx, item in enumerate(dataset):
            image = item['image'].unsqueeze(0).to(device)
            reference_caption = item['caption']

            # Generate caption
            generated_caption = generate_caption(model, tokenizer, image, device, output_dir=output_dir,
                                                  image_id=idx + 1)
            references.append([reference_caption.split()])
            hypotheses.append(generated_caption.split())

            # Calculate ROUGE scores
            scores = scorer.score(reference_caption, generated_caption)
            rouge_1_f1 += scores['rouge1'].fmeasure
            rouge_2_f1 += scores['rouge2'].fmeasure
            rouge_l_f1 += scores['rougeL'].fmeasure

            # Write captions to respective files
            with open(generated_captions_file, 'a', encoding='utf-8') as gen_file:
                gen_file.write(f"Sample {idx + 1}:\n{generated_caption}\n\n")
            with open(reference_captions_file, 'a', encoding='utf-8') as ref_file:
                ref_file.write(f"Sample {idx + 1}:\n{reference_caption}\n\n")

    n = len(references)
    rouge_1_f1 /= n
    rouge_2_f1 /= n
    rouge_l_f1 /= n

    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method1)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)

    # Calculate CIDEr and SPICE scores
    cider, _ = cider_scorer.compute_score(
        {idx: [' '.join(ref[0])] for idx, ref in enumerate(references)},
        {idx: [' '.join(hyp)] for idx, hyp in enumerate(hypotheses)}
    )

    try:
        spice, _ = spice_scorer.compute_score(
            {idx: [' '.join(ref[0])] for idx, ref in enumerate(references)},
            {idx: [' '.join(hyp)] for idx, hyp in enumerate(hypotheses)}
        )
    except Exception as e:
        print(f"SPICE evaluation skipped: {e}")
        spice = 0.0

    # Write evaluation metrics to file
    with open(metrics_file, 'a', encoding='utf-8') as metrics:
        metrics.write("Evaluation Metrics:\n")
        metrics.write("=" * 50 + "\n")
        metrics.write(f"BLEU-1: {bleu1:.4f}\n")
        metrics.write(f"BLEU-2: {bleu2:.4f}\n")
        metrics.write(f"BLEU-3: {bleu3:.4f}\n")
        metrics.write(f"BLEU-4: {bleu4:.4f}\n")
        metrics.write(f"ROUGE-1 F1: {rouge_1_f1:.4f}\n")
        metrics.write(f"ROUGE-2 F1: {rouge_2_f1:.4f}\n")
        metrics.write(f"ROUGE-L F1: {rouge_l_f1:.4f}\n")
        metrics.write(f"CIDEr: {cider:.4f}\n")
        metrics.write(f"SPICE: {spice:.4f}\n")

    print("\nEvaluation Metrics:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"ROUGE-1 F1: {rouge_1_f1:.4f}")
    print(f"ROUGE-2 F1: {rouge_2_f1:.4f}")
    print(f"ROUGE-L F1: {rouge_l_f1:.4f}")
    print(f"CIDEr: {cider:.4f}")
    print(f"SPICE: {spice:.4f}")

    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'ROUGE-1 F1': rouge_1_f1,
        'ROUGE-2 F1': rouge_2_f1,
        'ROUGE-L F1': rouge_l_f1,
        'CIDEr': cider,
        'SPICE': spice
    }

class MedicalImageCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_df, tokenizer, transform=None):
        self.image_dir = image_dir
        self.captions = captions_df
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.captions.iloc[idx, 1]
        caption = self.captions.iloc[idx, 8]
        img_path = next((f for f in os.listdir(self.image_dir) if f.startswith(str(img_name)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))), None)
        if img_path is None:
            print(f"Warning: Image file not found for {img_name} in {self.image_dir}")
            return None
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path).convert("L")  # Grayscale
        image = self.transform(image)
        tokenized = tokenizer(caption, return_tensors="pt", max_length=100, padding="max_length", truncation=True)
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption
        }

# --- Model Components ---
import torchvision.models as models
class MedicalVisionExtractor(nn.Module):
    def __init__(self):
        super(MedicalVisionExtractor, self).__init__()
        weights_path = r"C:\\Users\\Janath Nikash\\Desktop\\work_I\\weights\\resnet50-0676ba61.pth"
        resnet = models.resnet50()
        resnet.load_state_dict(torch.load(weights_path))
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, images):
        return self.feature_extractor(images)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads."

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(attention_output)
        return output, attention_weights

class CapVisNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feature_dim, hidden_dim, num_heads=8):
        super(CapVisNet, self).__init__()
        self.vision_extractor = MedicalVisionExtractor()
        self.global_projection = nn.Linear(feature_dim, hidden_dim)
        self.local_projection = nn.Linear(feature_dim, hidden_dim)
        self.multi_head_attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.language_generator = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=3, batch_first=True
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.vision_extractor(images)
        global_features = features.mean(dim=(2, 3))
        local_features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        projected_global_features = self.global_projection(global_features)
        projected_local_features = self.local_projection(local_features)
        attended_features, attention_weights = self.multi_head_attention(
            query=projected_global_features.unsqueeze(1),
            key=projected_local_features,
            value=projected_local_features
        )
        attended_features = attended_features.squeeze(1)
        hidden_state = attended_features.unsqueeze(0).repeat(self.language_generator.num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state)
        embeddings = self.embedding(captions)
        outputs, _ = self.language_generator(embeddings, (hidden_state, cell_state))
        logits = self.fc(outputs)
        return logits, attention_weights

class LanguageGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=3, dropout=0.3):
        super(LanguageGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, captions, hidden_state=None, cell_state=None):
        embeddings = self.embedding(captions)
        if hidden_state is None or cell_state is None:
            batch_size = captions.size(0)
            device = captions.device
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        lstm_out, (hidden_state, cell_state) = self.lstm(embeddings, (hidden_state, cell_state))
        logits = self.fc(lstm_out)
        logits = self.softmax(logits)
        return logits, hidden_state, cell_state

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
import torch

def preprocess_image(image_input, device):
    """
    Preprocesses an image for the model.

    Args:
        image_input: A PIL image, tensor, or file path (string).
        device: The device to move the tensor to.

    Returns:
        A preprocessed image tensor.
    """
    # If image_input is a file path, load it as a PIL image
    if isinstance(image_input, str):
        image_input = Image.open(image_input).convert("L")  # Convert to grayscale

    if isinstance(image_input, torch.Tensor):
        if image_input.dim() == 4:
            image_input = image_input.squeeze(0)
        image_input = ToPILImage()(image_input.cpu())
    if image_input.mode != 'L':
        image_input = image_input.convert('L')
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image_input).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

from gingerit.gingerit import GingerIt
import requests
from textblob import TextBlob

def clean_caption(caption):
    blob = TextBlob(caption)
    return str(blob.correct())

import torch.nn.functional as F

def collate_fn(batch):
    max_height = max(item['image'].shape[1] for item in batch)
    max_width = max(item['image'].shape[2] for item in batch)
    padded_images = []
    for item in batch:
        image = item['image']
        _, h, w = image.shape
        pad_h = max_height - h
        pad_w = max_width - w
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(padded_image)
    images = torch.stack(padded_images)
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    return {'image': images, 'input_ids': input_ids, 'attention_mask': attention_masks}

def clean_caption_preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9,.!? ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
import json

if __name__ == "__main__":
    with open('custom_vocab.json', 'r') as vocab_file:
        vocab = json.load(vocab_file)
    tokenizer_path = r"C:\\Users\\Janath Nikash\\Desktop\\work_I"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True
        )
        print("Tokenizer loaded successfully using AutoTokenizer.")
    except:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=rf"{tokenizer_path}\tokenizer.json"
        )
        tokenizer.pad_token = "[PAD]"
        tokenizer.cls_token = "[CLS]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.unk_token = "[UNK]"
        tokenizer.mask_token = "[MASK]"
        print("Tokenizer loaded successfully using PreTrainedTokenizerFast.")
    vocab_size = len(vocab)
    print(vocab_size)
    output_dir = "attention_maps"
    projections = pd.read_csv("C:\\Users\\Janath Nikash\\Desktop\\work_I\\archive\\indiana_projections.csv")
    reports = pd.read_csv("C:\\Users\\Janath Nikash\\Desktop\\work_I\\archive\\indiana_reports.csv")
    merged_data = pd.merge(projections, reports, on='uid').dropna()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_data = merged_data.rename(columns={"filename": "image_name", "findings": "caption"})
    merged_data["caption"] = merged_data["caption"].apply(clean_caption_preprocess)
    image_dir = "C:\\Users\\Janath Nikash\\Desktop\\work_I\\archive\\images\\images_normalized"
    dataset = MedicalImageCaptionDataset(image_dir, merged_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    model = CapVisNet(vocab_size=vocab_size, embedding_dim=256, feature_dim=2048, hidden_dim=512)
    print(device)
    model.embedding = nn.Embedding(vocab_size, model.embedding.embedding_dim)

    # Check if pre-trained model exists
    model_path = r"C:\Users\Janath Nikash\Desktop\work_I\trained_model.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print("No pre-trained model found. Training the model...")
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        train_model(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=10
        )
        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Load or initialize the set of processed image IDs
    processed_ids_file = os.path.join(output_dir, "processed_ids.txt")
    if os.path.exists(processed_ids_file):
        with open(processed_ids_file, "r") as f:
            processed_ids = set(map(int, f.read().splitlines()))
    else:
        processed_ids = set()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # evaluate_model(model, tokenizer, dataset, device, transform)
    caption = generate_caption(
        model=model,
        tokenizer=tokenizer,
        image=r"C:\Users\Janath Nikash\Desktop\work_I\71bl.jpg",
        device=device,
        output_dir=output_dir,
        image_id=9999,
        max_seq_len=20,
        beam_width=7,
        temperature=0.8,
        repetition_penalty=2.0,
        processed_ids=processed_ids
    )
    print(f"Generated Caption: {caption}")

    # # Save processed IDs after generating captions
    # with open(processed_ids_file, "w") as f:
    #     f.write("\n".join(map(str, processed_ids)))

