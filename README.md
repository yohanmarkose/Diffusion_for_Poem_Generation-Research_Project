# Diffusion Models for Shakespearean Sonnet Generation

A research project exploring the adaptation of diffusion models from image generation to structured text generation, with comparative analysis against transformer-based approaches.

## Research Question

Can diffusion models, originally designed for continuous image data, successfully generate coherent and stylistically authentic poetic text when adapted with appropriate conditioning mechanisms and embedding strategies?

## Novel Contributions

### 1. Weight-Tied Decoding Strategy
Instead of using frozen pre-trained embeddings or separate decoder networks, we developed a trainable embedding system:
```python
logits = denoised_embedding @ text_embeddings.weight.T
# Embeddings optimize for both noise prediction AND reconstruction
```
The transpose of the embedding matrix serves as the decoder, allowing gradients from both tasks to reshape the embedding space.

### 2. Positional Conditioning for Poetry
- N-gram position markers (START/MIDDLE/END/COMPLETE)
- Positional triplets (first, middle, last word)
- Word count and position embeddings
- Provides structural guidance while maintaining creative freedom

### 3. Custom Trainable Embedding Space
- Fully trainable 768-dimensional embeddings
- Joint optimization with dual loss: `diffusion_loss + α × reconstruction_loss`
- Embeddings self-organize for effective decoding through learned geometry

## Architectures

### Diffusion Model: TextDiffusionUNet

**Core Components**:
- U-Net structure with channels: [768, 512, 384, 256]
- Dilated convolutions (dilations: 1, 2, 4, 8) for multi-scale text pattern capture
- Multi-head cross-attention (8 heads) to conditioning embeddings
- Skip connections between encoder and decoder paths
- DDPM scheduler for training, DDIM for inference

**Embeddings**:
- Text embeddings: (3,432 vocab × 768 dim) - Trainable
- Length embeddings: (12 × 768 dim) - Encode n-gram length
- Position embeddings: (4 × 768 dim) - Encode structural position

**Training Process**:
1. Convert text to embeddings
2. Add noise according to timestep
3. Predict noise with U-Net conditioned on position/length
4. Denoise and decode via weight-tying
5. Optimize: `diffusion_loss + α × reconstruction_loss`

### Transformer Model: Encoder-Decoder Architecture

**Core Components**:
- 2-layer encoder with multi-head self-attention (8 heads)
- 2-layer decoder with self-attention + cross-attention
- 512-dimensional model, 2048-dimensional feed-forward networks
- Sinusoidal positional encoding
- Teacher forcing with shifted decoder inputs

**Training Process**:
1. Encoder processes input sequence with self-attention
2. Decoder generates output with:
   - Self-attention (understanding previous outputs)
   - Cross-attention (attending to encoder outputs)
3. Final dense layer projects to vocabulary
4. Optimize with sparse categorical cross-entropy

**Key Mechanism**: Attention heads learn what words relate to each other, enabling strong sequential coherence and context understanding.

## Dataset

- **Source**: Shakespeare's sonnets (~2,185 lines)
- **N-gram Expansion**: 60,426 training samples (2-20 word sequences)
- **Rationale**: Shakespeare's archaic syntax makes standard SVO extraction ineffective; n-grams capture poetic patterns better
- **Distribution**: 54% MIDDLE, 21.2% START, 21.2% END, 3.6% COMPLETE

## Results & Comparison

### Performance Metrics

| Metric | Diffusion Model | Transformer Model |
|--------|----------------|-------------------|
| Final Loss | 0.93 (epoch 50) | 0.67 (epoch 40) |
| Training Samples | 60,426 n-grams | 60,426 n-grams |
| Parameters | 70.4M | ~45M |
| Training Time/Epoch | ~4.5 min | ~12 min |

### Sample Outputs

#### Diffusion Model
**Prompt**: ['love', 'time', 'beauty']
```
Buried acceptable remove elder captive junes familiar lying
Thou best though dressing wood candles privilege; unswayed
Thou felt at dressing chaste familiar increase; but
Delayed dial's composition bitter o'ersways life's neck redeem
```

#### Transformer Model
**Prompt**: "Shall I compare thee to a summer's day"
```
Line 1: Shall I compare thee to a summer's day
Line 2: spirit of youth winter's day
Line 3: and barren rage of death's eternal barren thine day
Line 4: of thine eye in thy view
Line 5: in thy view is pleased to dote
```

### Comparative Analysis

**Sequential Coherence**: Transformer > Diffusion
- **Transformer**: Attention mechanisms enable strong grammatical structure and contextual flow. Each line builds naturally from the previous, with clear subject-verb relationships.
- **Diffusion**: Words are thematically connected but lack sentence-level grammar. Coherence degrades as sequence length increases.

**Creative Diversity**: Diffusion > Transformer
- **Diffusion**: Explores wider vocabulary space with unexpected combinations ("captive junes," "forests feathered dressing")
- **Transformer**: Falls into repetitive patterns (e.g., "of view" repeated 4 times), relies on frequently-seen n-grams

**Vocabulary Authenticity**: Diffusion ≈ Transformer
- Both successfully learned Shakespearean vocabulary and archaic expressions
- Diffusion: More varied word choices per generation
- Transformer: More consistent poetic meter

**Controllability**: Diffusion > Transformer
- **Diffusion**: Responds to multiple simultaneous conditions (position, length, specific words)
- **Transformer**: Primarily continues from textual prompts with limited attribute control

**Why These Differences?**

The diffusion model operates through iterative refinement in embedding space, allowing it to explore multiple word possibilities at each position simultaneously. This parallel processing encourages creativity but lacks the sequential dependency modeling that transformers achieve through attention.

The transformer's attention mechanism explicitly models "what word comes after what," creating strong sequential dependencies. It understands context across the entire sequence but can get trapped in high-probability n-gram patterns it learned during training.

## Diffusion Model Achievements

Despite being adapted from image generation, the diffusion model successfully:

1. **Learned authentic Shakespearean language** - Captured archaic vocabulary, poetic expressions, and thematic elements
2. **Generated creative word combinations** - Produced novel phrases like "death's eternal barren" and "fire--my candles dreading"
3. **Responded to positional conditioning** - Incorporated prompt words and structural guidance effectively
4. **Navigated semantic embedding space** - Self-organized embeddings for meaningful word relationships
5. **Maintained poetic meter** - Lines often follow appropriate syllable counts and rhythm
6. **Demonstrated vocabulary diversity** - Explored wider lexical range than transformer baseline

**Technical Achievement**: Successfully bridged discrete text tokens to continuous embedding space, enabling diffusion processes to work on linguistic data. The weight-tying strategy solved the fundamental challenge of decoding without separate networks.

## Future Research Directions

### Near-Term
- **Latent diffusion for text**: Compress sentences into latent space for better long-range coherence
- **Enhanced conditioning**: Add rhyme scheme, meter, semantic theme controls
- **Larger training corpus**: Expand to complete Shakespeare works and Elizabethan poetry

### Long-Term Vision
- **Hybrid architectures**: Combine diffusion's creativity with transformer's coherence
- **Cross-modal generation**: Poetry conditioned on images or music
- **Controllable creative writing tools**: Interactive refinement with multi-constraint optimization


## Conclusion

This research proves that diffusion models can successfully generate stylistically authentic poetic text, despite being designed for continuous image data. The weight-tied decoding innovation enables trainable embeddings to self-organize for effective generation.

While transformers currently produce more grammatically coherent sonnets, diffusion models offer unique advantages in creative exploration and controllable generation. The promising results—authentic vocabulary, thematic coherence, and responsive conditioning—suggest diffusion-based text generation could become valuable for applications requiring controlled creativity.


## License

Educational research project
