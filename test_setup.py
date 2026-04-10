"""
test_setup.py — Verify the project structure and imports work correctly.
Run this before training to make sure everything is set up right.

Usage:
    python test_setup.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def check(label, fn):
    try:
        fn()
        print(f"  ✅  {label}")
        return True
    except Exception as e:
        print(f"  ❌  {label}")
        print(f"       {e}")
        return False


def test_imports():
    import torch
    import torchvision
    import PIL
    import numpy
    print(f"     torch={torch.__version__}  torchvision={torchvision.__version__}")


def test_model_forward():
    import torch
    from models.vqa_model import VQAModel
    model = VQAModel(vocab_size=1000, num_answers=100, embed_dim=256)
    images    = torch.randn(2, 3, 224, 224)
    questions = torch.randint(0, 1000, (2, 20))
    out = model(images, questions)
    assert out.shape == (2, 100), f"Expected (2,100) got {out.shape}"


def test_tokenizer():
    from utils.tokenizer import VQATokenizer
    tok = VQATokenizer(max_len=10)
    tok.build_vocab(["What is in the image?", "How many dogs are there?", "Is it red?"], min_freq=1)
    ids = tok.encode("What is in the image?")
    assert len(ids) == 10


def test_transforms():
    from PIL import Image
    import numpy as np
    from utils.dataset import get_transform
    img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    t = get_transform("val")
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)


def test_helpers():
    import torch
    from utils.helpers import accuracy, top_k_accuracy
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    labels = torch.tensor([1, 0])
    assert accuracy(logits, labels) == 1.0


def main():
    print("\n" + "="*50)
    print("  VQA Project — Setup Test")
    print("="*50 + "\n")

    results = [
        check("Python imports (torch, torchvision, PIL)", test_imports),
        check("VQAModel forward pass",                    test_model_forward),
        check("VQATokenizer build + encode",              test_tokenizer),
        check("Image transforms (224×224)",               test_transforms),
        check("Helper functions (accuracy)",              test_helpers),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*50}")
    print(f"  {passed}/{total} tests passed")

    if passed == total:
        print("  🎉 All good! Ready to train.")
        print("\n  Next steps:")
        print("    1. python download_data.py       # download VQA v2")
        print("    2. python train.py --debug        # quick smoke test")
        print("    3. python train.py --epochs 30    # full training")
        print("    4. streamlit run app.py           # launch UI")
    else:
        print("  ⚠️  Fix the errors above before training.")
        print("  Install missing packages: pip install -r requirements.txt")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
