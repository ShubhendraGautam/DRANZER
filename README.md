# Attention in C - Neural Model with Multi-Head Attention

A complete transformer-based neural network implemented from scratch in C with **multi-head self-attention**, **layer normalization**, **gradient-based training**, and **advanced sampling strategies**.

## 🎯 Project Phases

### Phase 1: Attention Core
- ✅ Multi-head self-attention mechanism (4 parallel attention heads)
- ✅ Xavier weight initialization
- ✅ Residual connections
- ✅ Next-token prediction and inference

### Phase 2: Stability & Learning
- ✅ Layer normalization (learnable γ and β parameters)
- ✅ Adaptive learning rate scheduling
- ✅ Loss tracking and learning metrics
- ✅ Feedforward networks with ReLU activation

### Phase 3: Production & Scale
- ✅ Configuration file management (JSON-compatible)
- ✅ Training checkpoints (resumable training)
- ✅ 4 sampling strategies: greedy, top-k, top-p, beam search
- ✅ Batch processing infrastructure
- ✅ **Command-line interface for all modes**

## 📁 Project Structure

```
attention_in_c/
├── README.md                   # This file
├── Makefile                    # Top-level build coordinator
├── test.txt                    # Sample training data
│
├── libs/                       # Core utilities
│   ├── Makefile
│   ├── include/
│   │   ├── hashmap.h          # Type-aware hash table
│   │   ├── byte_pair_encoding.h
│   │   └── debug.h            # DEBUG_PRINT macro
│   └── src/
│       ├── hashmap.c
│       └── byte_pair_encoding.c
│
├── src/                        # Main application (Phase 1-3)
│   ├── Makefile
│   ├── main.c                 # 3 modes: train, infer, generate
│   ├── model.c/h              # Neural model + layer normalization
│   ├── tokenizer.c/h          # BPE tokenizer API
│   ├── attention.c/h          # Self-attention mechanism
│   ├── cli.c/h                # Command-line argument parsing
│   ├── config.c/h             # Configuration management
│   ├── sampling.c/h           # Sampling strategies
│   ├── batch.c/h              # Batch processing
│   ├── checkpoint.c/h         # Training checkpoints
│   ├── include/debug.h        # Debug utilities
│   └── app.out                # Compiled executable
│
├── config.json                 # Training configuration
├── dranzer.pth                 # Latest model weights
├── checkpoints/                # Training checkpoint history
│   ├── checkpoint_epoch_1_step_100.ckpt
│   └── config.txt
│
└── .agents.md, AGENTS.md, SKILL.md    # AI assistant documentation
```

## 🔧 Dependencies

- **Compiler**: clang (C11 support)
- **Libraries**: libc, libm (standard math)
- **Build**: GNU Make

### Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install clang make build-essential
```

**macOS:**
```bash
brew install clang make
```

## 🏗️ Building

### Quick Build

```bash
cd src
make
```

Produces: `src/app.out` (executable)

### Build Options

**Debug Mode** (enables DEBUG_PRINT statements):
```bash
make DEBUG=1
```

**AddressSanitizer** (memory error detection):
```bash
make ASAN=1
```

**Combined:**
```bash
make DEBUG=1 ASAN=1
```

**Clean Build Artifacts:**
```bash
make clean
```

## 🚀 Running the Program

The program supports 3 modes via **command-line interface**:

### 1. Training Mode (Default)

Train a new model on input data:

```bash
./app.out train --input data.txt --epochs 5
```

**Training Options:**
- `--input FILE` - Input training file (default: test.txt)
- `--epochs N` - Number of epochs (default: 1)
- `--batch-size N` - Batch size (default: 1)
- `--learning-rate LR` - Learning rate (default: 0.001)
- `--model FILE` - Model path to save (default: dranzer.pth)
- `--checkpoint-dir DIR` - Checkpoint directory (default: checkpoints)
- `--checkpoint-interval N` - Save checkpoint every N steps (default: 10)

**Examples:**

Basic training:
```bash
./app.out train --input corpus.txt
```

Training with hyperparameters:
```bash
./app.out train \
  --input corpus.txt \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.0005 \
  --checkpoint-interval 50
```

Training with GPU flag (if CUDA implemented):
```bash
./app.out train --input data.txt --gpu --debug
```

**Output Files Created:**
- `dranzer.pth` - Latest model weights
- `checkpoints/checkpoint_epoch_N_step_M.ckpt` - Intermediate checkpoints
- `checkpoints/config.txt` - Training configuration

### 2. Inference Mode

Run predictions on a prompt using a trained model:

```bash
./app.out infer --prompt "Once upon a time"
```

**Inference Options:**
- `--prompt TEXT` - Input prompt (**required**)
- `--model FILE` - Model to load (default: dranzer.pth)
- `--sampling STRATEGY` - Sampling: greedy, topk, topp (default: greedy)
- `--top-k N` - Top-k value (default: 5)
- `--top-p P` - Top-p nucleus threshold 0.0-1.0 (default: 0.9)

**Examples:**

Simple inference:
```bash
./app.out infer --prompt "hello"
```

With top-k sampling:
```bash
./app.out infer --prompt "The future is" --sampling topk --top-k 10
```

With top-p sampling:
```bash
./app.out infer --prompt "In a land" --sampling topp --top-p 0.95
```

### 3. Generation Mode

Generate text continuations from a seed prompt:

```bash
./app.out generate --prompt "hello" --length 100
```

**Generation Options:**
- `--prompt TEXT` - Seed prompt (**required**)
- `--model FILE` - Model to load (default: dranzer.pth)
- `--length N` - Tokens to generate (default: 50)
- `--sampling STRATEGY` - Sampling: greedy, topk, topp (default: greedy)
- `--top-k N` - Top-k value (default: 5)
- `--top-p P` - Top-p value 0.0-1.0 (default: 0.9)
- `--temperature T` - Temperature 0.0-2.0 (default: 0.8)

**Examples:**

Simple generation:
```bash
./app.out generate --prompt "Once upon a time" --length 100
```

Creative generation with sampling:
```bash
./app.out generate \
  --prompt "In the beginning" \
  --length 150 \
  --sampling topp \
  --top-p 0.9 \
  --temperature 0.85
```

### General Options (All Modes)

- `--gpu` - Enable GPU acceleration if available
- `--debug` - Enable debug output
- `--help` - Show comprehensive help

**Examples:**
```bash
./app.out --help
./app.out train --input data.txt --debug
./app.out generate --prompt "hello" --debug
```

## 📊 Quick Examples

### Complete Workflow: Train → Infer → Generate

```bash
# 1. Build
cd src
make
cd ..

# 2. Train model
./src/app.out train --input test.txt --epochs 5

# 3. Run inference
./src/app.out infer --prompt "test"

# 4. Generate text
./src/app.out generate --prompt "hello" --length 50 --sampling topk --top-k 5

# 5. Train with checkpoints (multiple training runs)
./src/app.out train --input test.txt --epochs 10 --checkpoint-interval 25

# 6. Debug mode (verbose output)
./src/app.out train --input test.txt --debug
```

### Model Files Explained

| File | Purpose | Created By |
|------|---------|-----------|
| `dranzer.pth` | **Latest model weights** (for inference) | Training mode |
| `checkpoints/checkpoint_epoch_N_*` | **Intermediate snapshots** (for resume/comparison) | Training mode |
| `checkpoints/config.txt` | **Training configuration** (hyperparameters) | Training mode |

**Key Difference:**
- **`dranzer.pth`** - Use for inference/generation (latest trained model)
- **`checkpoints/`** - Use for resuming training, comparing model quality, or rollback

## 🏛️ Architecture

### Model Components

```
Input Tokens
    ↓
BPE Tokenization (vocab_size=1000)
    ↓
Token Embedding (embedding_dim=64)
    ↓
Positional Encoding (sine/cosine)
    ↓
Multi-Head Attention (4 heads × 16 dim)
    ↓
Layer Normalization + Residual
    ↓
Feedforward Network (ReLU activation)
    ↓
Layer Normalization + Residual
    ↓
Output Projection
    ↓
Softmax + Next Token Prediction
```

### Key Features

- **Multi-Head Attention**: 4 parallel attention heads for diverse feature learning
- **Layer Normalization**: Stabilizes training with learnable gamma/beta parameters
- **Xavier Initialization**: Proper weight initialization for deep networks
- **Residual Connections**: Prevents vanishing gradient problem
- **Feedforward Networks**: Non-linear transformations between attention layers
- **Gradient Descent Training**: Backpropagation with cross-entropy loss
- **Advanced Sampling**: 4 strategies for diverse generation

## 📈 Model Hyperparameters

| Parameter | Default | Adjustable |
|-----------|---------|-----------|
| Vocabulary Size | 1000 | C code (recompile) |
| Embedding Dim | 64 | C code (recompile) |
| Attention Heads | 4 | C code (recompile) |
| Max Sequence | 512 | C code (recompile) |
| Learning Rate | 0.001 | `--learning-rate` ✅ |
| Batch Size | 1 | `--batch-size` ✅ |
| Epochs | 1 | `--epochs` ✅ |
| Checkpoint Interval | 10 | `--checkpoint-interval` ✅ |

**To Change Fixed Parameters**: Edit constants in [src/main.c](src/main.c) lines 23-26, then rebuild with `make`.

## 🔍 Debugging & Development

### Enable Debug Output

```bash
cd src
make DEBUG=1
cd ..
./src/app.out train --input test.txt --debug
```

Shows:
- BPE tokenization details
- Model initialization info
- Loss values during training
- Detailed step-by-step execution

### Memory Safety Checks

```bash
cd src
make ASAN=1
cd ..
./src/app.out train --input test.txt
```

Detects:
- Memory leaks
- Use-after-free bugs
- Out-of-bounds access
- Uninitialized memory

### Combined Debugging

```bash
cd src
make DEBUG=1 ASAN=1
cd ..
./src/app.out train --input test.txt --debug
```

## 📚 Build System Details

### Multi-Level Makefiles

**`Makefile`** (root): Delegates to subdirectories
**`libs/Makefile`**: Builds `libattention.a` containing:
- `hashmap.c` - Generic hash table with type-aware storage
- `byte_pair_encoding.c` - BPE tokenizer implementation

**`src/Makefile`**: Builds application:
- Compiles 9 source files (main, model, tokenizer, attention, cli, config, sampling, batch, checkpoint)
- Links against `libattention.a`
- Supports DEBUG and ASAN flags

### Compilation Pipeline

```
source files (.c)
    ↓
[cc -c] object files (.o)
    ↓
[ar rcs] libattention.a (libs)
    ↓
[cc link] app.out executable
```

## 🎓 Learning Resources

### Understanding the Code

- **[.agents.md](.agents.md)** - 8 AI agent definitions for code navigation
- **[AGENTS.md](AGENTS.md)** - Human-readable agent workflows
- **[SKILL.md](SKILL.md)** - 9 technical skill domains with patterns

### File Dependencies

```
main.c
├── model.c/h         (neural model)
├── tokenizer.c/h     (BPE encoding)
├── attention.c/h     (self-attention)
├── cli.c/h           (argument parsing)
├── config.c/h        (configuration)
├── sampling.c/h      (sampling strategies)
├── batch.c/h         (batch processing)
└── checkpoint.c/h    (checkpoints)
    └── byte_pair_encoding.c/h (in libs)
```

## 🔬 Phase 3 Output Example

```
>>> Mode: TRAIN

=== Neural Model Training ===

[1] Creating BPE encoder and tokenizing input...
   Building BPE vocabulary...
   Vocabulary size: 269 tokens
   Tokenized into 101 tokens

[2] Initializing neural model...
   Model initialized:
   - Vocabulary: 1000 tokens
   - Embedding dim: 64
   - Attention heads: 4

[3] Training model...
   Epochs: 1, Batch size: 1, Learning rate: 0.00100000
   Epoch 1/1 - Loss: 7.216766
   Training steps: 100

[4] Demonstrations...
   ✓ Attention mechanism validated
   ✓ Sampling strategies demonstrated
   ✓ Batch infrastructure ready

[5] Next token prediction...
   Predicted next token ID: 51
   ✓ Inference working

[6] Saving model to dranzer.pth
   ✓ Model saved

[7] Saving configuration...
   ✓ Config saved to checkpoints/config.txt

[8] Creating checkpoint...
   ✓ Checkpoint saved: checkpoints/checkpoint_epoch_1_step_100.ckpt

=== Summary ===
✓ BPE tokenization:     101 tokens
✓ Training:             100 steps, loss 7.216766
✓ Model saved:          dranzer.pth
✓ Config saved:         checkpoints/config.txt
```

## 🚦 Troubleshooting

**Build fails with `undefined reference`**
- Run `make clean` then `make` from `src/` directory
- Ensure libraries are built first

**Model not found for inference**
- Train first: `./app.out train --input data.txt`
- This creates `dranzer.pth`

**Memory issues during training**
- Reduce batch size: `--batch-size 1`
- Use AddressSanitizer to detect leaks: `make ASAN=1`

**Slow compilation**
- Recompile only changed files: `make` (from `src/`)
- Avoid `make clean` unless necessary

## 📄 License

MIT License - Free for educational and research use.

## 👤 Author

Built as a comprehensive neural network implementation in C, covering:
- Core ML concepts (attention, normalization, sampling)
- Low-level systems programming (memory management, binary formats)
- Software engineering (modular design, testing, documentation)

---

**Last Updated**: April 2026
**Status**: Phase 3 Complete ✅
**Version**: 1.0 (CLI-Ready)
