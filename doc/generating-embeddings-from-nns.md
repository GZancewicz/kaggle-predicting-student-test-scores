# Generating Embeddings from Neural Networks

## The Problem

When we have categorical features like `course = "b.tech"`, we typically convert to numbers:

```
b.tech  → 0
b.sc    → 1
diploma → 2
...
```

But `0` is just an arbitrary ID. The model doesn't know anything about what b.tech *means* or how it relates to other courses.

## The Embedding Solution

Instead of a single number, we represent each category as a **vector of learnable numbers**:

```
course 0 (b.tech):  [?, ?, ?, ?]   ← 4 numbers to be learned
course 1 (b.sc):    [?, ?, ?, ?]
course 2 (diploma): [?, ?, ?, ?]
...
```

When the model sees `course = 0`, it looks up row 0 and gets those 4 numbers.

## How Embeddings Are Learned

They're learned **during training**, not before. The process:

1. **Start**: Fill embedding table with random numbers
2. **Train**: Model makes predictions, gets them wrong
3. **Update**: Backpropagation adjusts the embedding values (along with all other weights)
4. **Repeat**: After many iterations, the values become meaningful

## What "Meaningful" Means

After training:
```
b.tech:  [0.8, 0.2, ...]
bca:     [0.7, 0.3, ...]   ← similar vectors (both STEM)
ba:      [-0.5, 0.9, ...]  ← different vector (arts)
```

The model discovers that b.tech and bca should have similar embeddings because students in those courses behave similarly with respect to the prediction target.

## PyTorch Implementation

### Defining the Embedding Layer

```python
import torch.nn as nn

# Create embedding: 7 categories, 4 dimensions each
self.course_embedding = nn.Embedding(7, 4)
```

This creates a 7×4 matrix filled with random values.

### What nn.Embedding Actually Is

It's just a **lookup table**:

```python
class Embedding:
    def __init__(self, num_rows, num_cols):
        self.weight = random_matrix(num_rows, num_cols)  # learnable

    def forward(self, indices):
        return self.weight[indices]  # just indexing!
```

No complex math - just `matrix[row_index]` returns that row.

### Forward Pass

```python
input = torch.tensor([0, 2, 1, 0])  # batch of course indices

output = self.course_embedding(input)
# Looks up rows 0, 2, 1, 0
# Returns four 4-dimensional vectors
# Shape: (4, 4)
```

### What Gets Updated During Training

All learnable parameters are updated together:

```
model.parameters() = [
    dense1.weight,              # updated
    dense1.bias,                # updated
    course_embedding.weight,    # updated (the lookup table!)
    method_embedding.weight,    # updated
    ...
]
```

The embedding table is just another matrix of numbers that gets adjusted via backpropagation.

## Choosing Embedding Dimensions

There's no perfect answer. Common approaches:

| Categories | Typical Dim |
|------------|-------------|
| 2-3 | 2 |
| 5-7 | 3-4 |
| 10-20 | 5-10 |
| 100+ | 20-50 |

**Rules of thumb**:
- `embedding_dim = min(50, (num_categories + 1) // 2)`
- FastAI formula: `min(600, round(1.6 * num_categories^0.56))`

In practice, try a few values and see what works best.

## Extracting Embeddings After Training

Once trained, you can extract the learned vectors:

```python
# Get the learned embedding matrix
course_vectors = model.course_embedding.weight.data.cpu().numpy()

# Result: 7×4 matrix
# course_vectors[0] = embedding for b.tech
# course_vectors[1] = embedding for b.sc
# ...
```

## Using Embeddings as Features for Other Models

The key insight: **embeddings learned by a NN can be used as features in gradient boosting**.

### Step 1: Train NN with Embeddings

```
Train NN → learn embedding tables

course_embedding after training:
  b.tech  → [0.8, 0.2, -0.3, 0.5]
  b.sc    → [0.7, 0.3, -0.2, 0.4]
  diploma → [-0.5, 0.1, 0.8, -0.2]
```

### Step 2: Transform Data Using Learned Embeddings

```
Original row:
  age=20, course="b.tech", study_hours=5, ...

Becomes:
  age=20,
  course_emb_0=0.8, course_emb_1=0.2, course_emb_2=-0.3, course_emb_3=0.5,
  study_hours=5, ...
```

### Step 3: Train LightGBM on Expanded Features

```python
# Replace category indices with their embeddings
X_train_expanded = add_embeddings(X_train, course_vectors, method_vectors, ...)

# Train gradient boosting
lgb_model.fit(X_train_expanded, y_train)
```

## Why This Helps

- **NN learns relationships** between categories that we didn't specify
- **LightGBM is better at tabular prediction** than neural nets
- **Combining both** gives the best of both worlds

This is a common Kaggle technique for competitions with categorical features.

## Summary

1. **Embeddings** = learnable lookup table mapping category → vector
2. **Learned during training** alongside all other weights
3. **After training**, extract and use as features in other models
4. **Dimension** is a hyperparameter to tune (start with 2-8 for small categories)
