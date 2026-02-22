# Food Image Classification with CNN & EfficientNet

---

## Business Problem

As food services move to digital platforms, accurate food image recognition becomes critical for:

- Food delivery automation
- Smart menu systems
- Dietary tracking
- Cloud kitchens

Misclassification (e.g., *carbonara vs. bolognese*) leads to:

- Wrong orders  
- Refund costs  
- Customer dissatisfaction  

Our goal:  
> Build a high-accuracy food classification system using CNN-based deep learning.

---

## Dataset

**[Food-101](https://huggingface.co/datasets/ethz/food101)**

- 101 food categories  
- 101,000 images  
- 750 train / 250 test per class  
- Real-world noise & high visual similarity  

Challenges:

- High intra-class variation  
- High inter-class similarity  
- Variable image sizes  

---

## Exploratory Data Analysis

Key findings:

- Most images: `512×512`
- Strong visual overlap between categories (e.g., soups, salads)
- Natural hierarchy:
  - Appetizer
  - Main Course
  - Dessert
 
→ Motivated our **two-stage hierarchical model**

---

## Models

### 1️⃣ Custom CNN

#### One-stage
- 2 Conv layers → FC → 101 classes  
- Accuracy ≈ **9%**

#### Two-stage (Hierarchical)
Stage 1 → coarse class  
Stage 2 → fine-grained class  

Improvements:

- More structured errors
- Better feature localization

But:

- Max accuracy ≈ **14.1%**

---

### 2️⃣ Transfer Learning — EfficientNet-B0

Why EfficientNet-B0?

- Best speed–accuracy tradeoff
- Pretrained on ImageNet
- Compound scaling

Our setup:

- Replaced classification head → 101 classes
- Full fine-tuning (no frozen layers)

---

## Results

### Custom CNN

| Model | Performance |
|-------|------------|
One-stage | Severe misclassification |
Two-stage | More meaningful confusion patterns |

---

### EfficientNet-B0

| Model | Test Accuracy |
|-------|--------------|
One-stage | **58.27%** |
Two-stage | 46.37% |

Observations:

- One-stage > Two-stage
- Two-stage suffered from:
  - subclass imbalance
  - overfitting

---

## Key Insights

### Why transfer learning wins

Pretrained models already understand:

- edges
- textures
- shapes

Custom CNN must learn from scratch → data & compute insufficient.

---

### Why hierarchical failed for EfficientNet

Although theoretically strong:

- error propagation between stages
- small subclass sample size
- overfitting in fine-grained classifiers

---

## Limitations

- EfficientNet trained on subset (compute constraint)
- Models optimized independently
- Overfitting (train ≈ 99%, test 40–60%)

---

## Future Work

- Freeze early EfficientNet layers
- Class-balanced sampling
- Label smoothing
- Attention mechanisms
- Train all models on full dataset
- End-to-end hierarchical learning

---

## Takeaways

- Transfer learning is essential for fine-grained vision tasks  
- Hierarchical design helps error interpretability, but increases training complexity  

---

## References

- Food-101 (Bossard et al., 2014)
- DeepFood (2016)
- EfficientNet (Tan & Le, 2019)
- Transfer learning for food classification (Singh & Susan, 2023)
- Hierarchical food classification (Pan et al., 2023)

