# CS490 LoRA and vLLM Tutorial
**Author:** Alice Chua

A tutorial on fine-tuning LoRA adapters and efficiently serving them using vLLM.

---

## Overview

In this tutorial, we work through the lifecycle of a modern LLM project:
1.  **Fine-Tuning (LoRA):** We use **Unsloth** to fine-tune Meta's **Llama 3 (8B)** model using Low-Rank Adaptation (LoRA).
2.  **Multi-Task Training:** We train two distinct adapters on the same base model:
    * **SQL Adapter:** Converts natural language questions into SQL queries.
    * **Shakespeare Adapter:** Generates text in the style of Shakespeare.
3.  **Deployment (vLLM):** We deploy a high-performance **vLLM** inference server that serves *both* adapters simultaneously using Multi-LoRA support.

---

## Files
* `020226_tutorial_lora_vllm.ipynb`: The main Jupyter Notebook containing all code for training, saving, and serving the models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/drive/1kXWnzejla9Lx3kiIsQBd8mP-g5qw48jx?usp=sharing)



---

## Prerequisites
To run this tutorial, you will need:
* **Hugging Face Account:** You need an access token with read permissions to download Llama 3.
* **GPU:** A GPU with at least 12GB VRAM (NVIDIA T4 or better).

---

## How to Run

### Option 1: Google Colab (Recommended)
This notebook is optimized to run on the free tier of Google Colab.

1.  Upload the notebook to Google Colab.
2.  **Important:** Change the Runtime Type to GPU.
    * Go to `Runtime` -> `Change runtime type`.
    * Select **T4 GPU**.
3.  **Add Secrets:**
    * Click the "Keys" icon in the left sidebar.
    * Add a secret named `HF_TOKEN` containing your Hugging Face API key.
    * Enable "Notebook Access" for the secret.
4.  Run all cells in order.

### Option 2: Run Locally
If you have a local machine with an NVIDIA GPU (RTX 3060 12GB or higher) and Linux/WSL:

1.  Clone this repository.
2.  Install the required libraries:
    ```bash
    pip install unsloth "unsloth[colab-new]" @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)
    pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
    pip install vllm
    ```
3.  Launch Jupyter Lab or Notebook:
    ```bash
    jupyter lab
    ```
---

## Technologies Used
* **[Unsloth](https://github.com/unslothai/unsloth):** Used for 2x faster training and 60% less memory usage.
* **[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685):** Parameter-efficient fine-tuning technique.
* **[vLLM](https://github.com/vllm-project/vllm):** High-throughput serving engine with PagedAttention and Multi-LoRA support.

## Reference
* [vLLM on Google Colab](https://medium.com/@hakimnaufal/trying-out-vllm-deepseek-r1-in-google-colab-a-quick-guide-a4fe682b8665)
