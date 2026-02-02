## 🛠️ Setup Instructions
1. Prerequisites
Ensure you have an NVIDIA GPU and Python 3.10+. This setup uses bitsandbytes for quantization and xformers for efficient attention.

2. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: This project requires specific versions of trl (version < 0.9.0) and peft compatible with Unsloth and vLLM multi-LoRA integration.

## 📖 Usage Guide

### Step 1: Fine-Tuning the Adapters
Run the training script to generate the LoRA weights. The script is designed to fine-tune two separate adapters sequentially, clearing GPU memory between sessions to prevent **Out-of-Memory (OOM)** errors.

```bash
python train.py
```

**Outputs:**

* **`./sql_adapter`**: Weights for the SQL generation task.
* **`./creative_adapter`**: Weights for the Shakespearean style task.

### Step 2: Start the vLLM Server

The server loads the base 4-bit model and dynamically maps the saved adapters to specific model IDs.

```bash
chmod +x serve.sh
./serve.sh
```

* **Endpoint**: `http://localhost:8000`
* **Quantization**: Uses `bitsandbytes` to ensure the 8B model fits on consumer-grade hardware.

### Step 3: Run Inference

Query the running server using the OpenAI-compatible client. This script demonstrates how to switch between models by changing the `model` ID in the request.

```bash
python client.py
```

---

## 📂 File Structure

| File | Description |
| --- | --- |
| `train.py` | Fine-tuning logic for SQL and Shakespeare adapters using Unsloth. |
| `serve.sh` | Shell script to launch the vLLM multi-LoRA server. |
| `client.py` | Python client for testing specific adapter endpoints. |
| `requirements.txt` | Necessary Python packages including `vllm`, `unsloth`, and `bitsandbytes`. |

---

## ⚙️ Configuration Details

### vLLM Server Arguments (`serve.sh`)

* **`--enable-lora`**: Enables the dynamic LoRA loading subsystem.
* **`--lora-modules`**: Maps custom names (e.g., `sql-model`) to local adapter paths.
* **`--max-loras`**: Sets the maximum number of adapters that can be active concurrently.
* **`--gpu-memory-utilization 0.9`**: Reserves 90% of VRAM for the KV cache and model weights.

### Training Hyperparameters (`train.py`)

* **Rank (r)**: 16
* **Target Modules**: All linear layers (`q, k, v, o, gate, up, down`) for maximum expressivity.
* **Batch Size**: 2 (with 4 Gradient Accumulation steps).
* **Max Steps**: 30 (Optimized for rapid demonstration/testing).