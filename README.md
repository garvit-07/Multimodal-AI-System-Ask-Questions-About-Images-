# VQA — Visual Question Answering (PyTorch + Streamlit)

A full Multimodal AI system: upload an image, ask a question, get an AI answer.

---

## Project Structure

```
vqa_project/
├── data/                   ← put downloaded dataset here
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_mscoco_val2014_annotations.json
│   └── images/
│       ├── train2014/      ← COCO train images
│       └── val2014/        ← COCO val images
├── models/
│   ├── vqa_model.py        ← VQA model definition
│   └── encoders.py         ← Image + Text encoders
├── utils/
│   ├── dataset.py          ← VQAv2 Dataset class
│   ├── tokenizer.py        ← Question tokenizer
│   └── helpers.py          ← Utility functions
├── checkpoints/            ← saved model weights go here
├── sample_images/          ← test images for Streamlit demo
├── train.py                ← training script
├── evaluate.py             ← evaluation script
├── inference.py            ← single-image inference
├── app.py                  ← Streamlit web UI
├── download_data.py        ← helper to download dataset
├── requirements.txt
└── README.md
```

---

## Dataset — VQA v2 (Official)

**Website:** https://visualqa.org/download.html

### Download these files:

#### Questions
```bash
# Train questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip

# Val questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
```

#### Annotations (Answers)
```bash
# Train annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

# Val annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
```

#### COCO Images
```bash
# Train images (~13 GB)
wget http://images.cocodataset.org/zips/train2014.zip

# Val images (~6 GB)
wget http://images.cocodataset.org/zips/val2014.zip
```

### Or run the helper:
```bash
python download_data.py
```

### Extract all zips into `data/`:
```bash
unzip "*.zip" -d data/
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Quick test (100 samples, 2 epochs)
python train.py --epochs 2 --batch_size 32 --max_samples 100 --debug

# Full training
python train.py --epochs 30 --batch_size 64 --lr 1e-4
```

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## Inference (CLI)

```bash
python inference.py --image sample_images/dog.jpg --question "What animal is in the image?"
```

---

## Streamlit App

```bash
streamlit run app.py
```

Then open http://localhost:8501

---

## Model Architecture

```
Image  ──► ResNet-50 ──► Linear(2048→512) ──┐
                                              ├──► Concat(1024) ──► MLP ──► Answer
Question ──► Embedding+LSTM ──► Linear(→512) ─┘
```

For better accuracy, set `--use_bert` to swap LSTM for BERT encoder.

---

## Notes

- Training on full VQAv2 needs a GPU with ~8GB VRAM
- In `--debug` mode it uses only 500 samples (CPU-friendly)
- The Streamlit app works with a trained checkpoint OR in demo mode with a pretrained CLIP model

## My Contributions
Fine-tuned the multimodal VQA model on VQA v2 dataset for improved answer accuracy and generalization
Implemented text encoding using BERT, replacing LSTM with contextual embeddings (--use_bert)
Designed and optimized image-text fusion pipeline (feature projection + concatenation + MLP)
Improved training pipeline with hyperparameter tuning (LR, batch size, epochs) and efficient batching
Built modular pipeline (dataset, tokenizer, encoders, inference) for scalability and experimentation
Developed end-to-end system including CLI inference and Streamlit-based web UI
Integrated pretrained vision backbone (ResNet-50) and applied transfer learning for feature extraction
