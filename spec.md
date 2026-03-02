# Lexical System: A Formal Documentation

## 1. Philosophical Foundation

Octopinion is a constructed language designed for a species of intelligent octopus. Its lexical system emerges from three fundamental constraints:

1. **Biological**: The language is signed using eight arms, suckers, mantle position, and siphon orientation. This imposes a natural limit on the number of primitive signs.
2. **Cognitive**: Octopus intelligence is decentralised and associative. Words should be compositional—their meaning arises from the superposition of simpler semantic primitives.
3. **Mathematical**: The culture uses an octal number system. This permeates the language's structure, including its inventory of primitive signs and its hierarchical organisation.

The lexical system documented here governs how **words** are formed from **syllables**. The syntactic system (cases, word order) is documented separately.

---

## 2. Core Axioms

### Axiom 1: The Syllabary

There exists a fixed, finite set of **primitive syllables** $\mathcal{S} = \{s_1, s_2, \dots, s_n\}$ where 26 (or any number congruent with octal thinking, e.g., 8, 16, 64). Each syllable corresponds to a unique physical sign involving arm position, sucker configuration, and optionally mantle posture.

### Axiom 2: Semantic Vectors

Every syllable $s_i$ is associated with a **semantic vector** $\mathbf{v}_i \in \mathbb{R}^d$, where $d$ is the dimensionality of the semantic space (typically 384 or 768). These vectors are unit-normalized: $\|\mathbf{v}_i\| = 1$.

The set $\{\mathbf{v}i\}{i=1}^n$ is the **codebook** of the language.

### Axiom 3: Word Composition

A word $W$ is a finite sequence of syllables $\langle s_{i_1}, s_{i_2}, \dots, s_{i_k}\rangle$. Its meaning is the weighted sum of their semantic vectors:

$$
\text{meaning}(W) = \sum_{j=1}^{k} \lambda^{j-1} \mathbf{v}_{i_j}
$$

where $\lambda \in (0,1)$ is the **decay factor**, a language-wide constant. The default value is $\lambda = 0.5$ (or $1\over8$ for strict octal alignment).

### Axiom 4: Order Matters

Because $\lambda^{j-1}$ strictly decreases with $j$, the position of a syllable in the sequence determines its contribution to the overall meaning. The first syllable contributes most; each subsequent syllable contributes less.

### Axiom 5: Case Is Separate

Case markers are **not** part of the lexical word. They are realised through the changes in the skin color.

This ensures the lexical system remains pure—words are bundles of semantic content only. Syntax is superimposed via separate channels.

---

## 3. The Semantic Space

The semantic space is a high-dimensional vector space where concepts are embedded. This space is **learned** from data, not manually defined. Its dimensions are latent and emerge from training on a corpus relevant to octopus experience.

### 3.1 Origin of the Space

The space is derived from a neural embedding model (e.g., BGE, Qwen Embedding) that has been trained on a large corpus of text. However, to make it octopus-relevant, the model may be **fine-tuned** or **adapted** using:

- A corpus of marine biology texts
- Simulated octopus experience data
- Contrastive learning with octopus-like perceptual pairs (e.g., smooth/spiky, prey/predator)

The resulting embedding space has $d$ dimensions (typically 384, 768, or 1024), each representing a latent feature that the model has learned to distinguish. These features may not be directly interpretable as human categories (like "edibility") but capture statistical regularities in the data.

### 3.2 Why Latent Dimensions?

The choice of learned, latent dimensions is deliberate:

- It avoids imposing human categories on an alien mind.
- It allows the semantic space to reflect true octopus-relevant distinctions, even if those distinctions are unfamiliar to us.
- It provides a rich, high-dimensional representation that can capture nuance.

---

## 4. Word Formation: The Encoding Algorithm

Given a target concept with semantic vector $\mathbf{t} \in \mathbb{R}^d$ (in the learned semantic space), the language produces a word—a sequence of syllables—through a **greedy residual pursuit** algorithm.

### 4.1 The Algorithm

```
function encode(target_vector t):
    residual = t
    sequence = []
    step = 0
    while norm(residual) > threshold and step < max_steps:
        # Find the syllable that best reduces the residual
        # The decay factor λ^(step) scales the contribution
        # but cancels out in the argmax
        scores = [dot(residual, v_i) for v_i in codebook]
        best_index = argmax(scores)

        # Add to sequence
        sequence.append(best_index)

        # Update residual: subtract the contribution of this syllable
        # scaled by the positional weight
        contribution = λ^step * codebook[best_index]
        residual = residual - contribution

        step += 1

    return sequence
```

### 4.2 Why Greedy Works

At each step, we choose the syllable that maximises $\mathbf{v}_i \cdot \mathbf{r}$, where $\mathbf{r}$ is the current residual. This minimizses $\|\mathbf{r} - \lambda^{\text{step}} \mathbf{v}_i\|^2$ because:

$$
\|\mathbf{r} - \lambda^{\text{step}} \mathbf{v}_i\|^2 = \|\mathbf{r}\|^2 + {\lambda^{\text{step}}}^2 - 2\lambda^{\text{step}} (\mathbf{v}_i \cdot \mathbf{r})
$$

Since $\lambda^{\text{step}}$ is positive and constant for the given step, maximising the dot product minimises the norm.

### 4.3 Termination

The algorithm stops when:

- The residual norm falls below a threshold $\epsilon$ (indicating the word adequately captures the concept).
- The maximum word length $L_{\max}$ is reached (typically 5–8 syllables, as longer words would be unwieldy to sign).

If terminated by length limit, the word is an approximation of the target concept. This is acceptable—language often has fuzzy boundaries.

---

## 5. Decoding: From Word to Meaning

Given a word $W = \langle s_{i_1}, s_{i_2}, \dots, s_{i_k}\rangle$, its meaning is computed deterministically:

$$
\text{decode}(W) = \sum_{j=1}^{k} \lambda^{j-1} \mathbf{v}_{i_j}
$$

This is a **linear composition**. The result is a point in semantic space that approximates some concept.

### 5.1 Similarity and Interpretation

To find the nearest "gloss" in a human language, we compute cosine similarity between the decoded vector and a database of concept embeddings (e.g., from the same embedding model). The closest match is the translation.

For example:

- $\text{decode}(\langle 3, 7, 2\rangle) \approx \text{embedding}(\text{"fish"})$
- $\text{decode}(\langle 1, 4\rangle) \approx \text{embedding}(\text{"hunt"})$

Because the semantic space is learned, these mappings are emergent and may not align perfectly with English categories—but they will be consistent within the language.

---

## 6. Learning the Codebook

The codebook $\{\mathbf{v}_i\}$ is not arbitrary. It is **learned** from data to optimise the encoding efficiency of the language.

### 6.1 Training Data

We start with a corpus of concepts relevant to octopus life. This corpus can be:

- A list of words with their embeddings from a pre-trained embedding model (e.g., BGE).
- A set of octopus-relevant concept pairs for contrastive learning.
- Simulated perceptual experiences represented as vectors.

The key is that the semantic space is already fixed (from a pre-trained model). The codebook learning operates within that space.

### 6.2 Differentiable Training with Gumbel-Softmax

To make the codebook learnable, we relax the discrete selection process using **Gumbel-Softmax**. For each training concept $\mathbf{t}$:

1. Initialise residual $\mathbf{r} = \mathbf{t}$.
2. For step $k = 0$ to $K-1$ (where $K$ is a fixed maximum, e.g., 4):
    - Compute logits: $\text{logits}_i = \mathbf{v}_i \cdot \mathbf{r}$ for all $i$.
    - Sample a one-hot vector $\mathbf{g}^{(k)} \sim \text{Gumbel-Softmax}(\text{logits}, \tau)$, where $\tau$ is temperature.
    - Compute contribution: $\mathbf{c}^{(k)} = \lambda^k \sum_i g_i^{(k)} \mathbf{v}_i$.
    - Update residual: $\mathbf{r} \leftarrow \mathbf{r} - \mathbf{c}^{(k)}$.
    - Store $\mathbf{g}^{(k)}$.
3. After $K$ steps, compute reconstruction:
    
    $$
    \hat{\mathbf{t}} = \sum_{k=0}^{K-1} \lambda^k \sum_i g_i^{(k)} \mathbf{v}_i
    $$
    
4. Loss: $\mathcal{L} = \|\mathbf{t} - \hat{\mathbf{t}}\|^2 + \beta \cdot \text{entropy\_penalty}$ (optional, to encourage peaky distributions).

Back-propagate through the unrolled steps to update $\mathbf{v}_i$. Anneal $\tau$ from high (e.g., 5) to low (e.g., 0.1) over training.

### 6.3 Post-Training Discretisation

After training, we fix the codebook. For inference (actual language use), we replace the Gumbel-Softmax with **argmax** at each step, yielding a deterministic discrete sequence.

---

## 7. The Decay Factor $\lambda$

The decay factor is a fundamental parameter of the language. Its value shapes how meaning is distributed across syllables.

### 7.1 Interpretation

- **High** $\lambda$ **(close to 1)**: Later syllables contribute almost as much as early ones. Words become more "flat"—order matters less, and longer words are more expressive.
    - E.g. used when chatting/formal writings so nuanced details can be reflected.
- **Low** $\lambda$ **(close to 0)**: The first syllable dominates; later syllables add only minor refinements. This creates a strong hierarchy: first syllable = broad category, subsequent = finer details.
    - E.g. to report the emergency

### 7.2 Default Value

For Octopinion, we set $\lambda = 0.5$ as the default. This provides a balance:

- First syllable contributes 50% of the total magnitude (if unit vectors).
- Second: 25%
- Third: 12.5%
- Fourth: 6.25%

Thus, words longer than 4–5 syllables add negligible precision.

### 7.3 Learning $\lambda$

Optionally, $\lambda$ can be treated as a learnable parameter during codebook training, allowing the language to find its optimal decay rate.

---

## 8. Hierarchical Interpretation

Although the semantic space's dimensions are latent and not manually labeled, the exponential decay naturally induces a **category–subcategory** hierarchy based on **position**, not on predefined axes:

| Position | Role | Contribution |
| --- | --- | --- |
| Syllable 1 | **Primary semantic component** | Dominates meaning; roughly corresponds to broad domain |
| Syllable 2 | **Secondary refinement** | Modifies and narrows the primary |
| Syllable 3 | **Tertiary detail** | Adds finer distinctions |
| Syllable 4 | **Quaternary nuance** | Minor adjustments |
| Syllable 5+ | **Marginal specificity** | Rarely used; near-threshold adjustments |

The actual semantic content of each position is determined by the learned codebook and the statistics of the training data. For example, if the training data has clusters of concepts (like marine animals, actions, qualities), the codebook may learn to allocate certain syllables to those clusters, and the greedy algorithm will tend to pick the cluster centroid first, then refine.

Thus, the hierarchy is **emergent**, not prescribed. This is more faithful to how a natural language might evolve.

### 8.1 Example

Suppose after training, the codebook has the following approximate associations (purely illustrative; actual vectors are high-dimensional but maybe interpretable by using the axis’s embedding to retrieve corresponding text in embedding space):

| Axis | Name | Description | Polarity |
| --- | --- | --- | --- |
| 1 | **Domain** | Entity existence | Abstract ↔ Concrete |
| 2 | **Texture** | Tactile quality | Smooth ↔ Spiky |
| 3 | **Pressure** | Water movement | Still ↔ Turbulent |
| 4 | **Chromatic** | Color and pattern visibility | Cryptic ↔ Conspicuous |
| 5 | **Temporal** | Duration and rhythm | Instant ↔ Cyclic |
| 6 | **Social** | Relation to conspecifics | Solitary ↔ Interactive |
| 7 | **Agency** | Capacity for independent motion | Inert ↔ Purposeful |
| 8 | **Metabolic** | Relevance to internal state | Neutral ↔ Urgent |
| … |  |  |  |

The meaning vector is the weighted sum.

- Syllable 1 (Domain): **Entity**
- Syllable 2 (Category): **Marine Animal**
- Syllable 3 (Subcategory): **Fish**
- Syllable 4 (Modifier): **Spiky**
- Syllable 5 (Modifier): **Dangerous**

The meaning vector is:

$$
\lambda = \lambda^0 s_1 + \lambda^1 s_7 + \lambda^2 s_3 + \lambda^3 s_{12} + \lambda^4 s_9 \\
\text{Something = Entity + Marine Animal + Fish + Spiky + Dangerous}
$$

where $\lambda$ is the embedding vector of word “pufferfish” in the space of embedding model.

---

## 9. Vocabulary Generation

Given a trained codebook and decay factor, we can generate the entire lexicon by:

1. Collecting a list of concepts relevant to the culture. Each concept is represented by its embedding vector in the semantic space.
2. For each concept, run the greedy encoding algorithm to obtain a syllable sequence.
3. Store the mapping: concept ↔ sequence.

### 9.1 Handling Synonyms

Different sequences may produce similar meaning vectors. These are **synonyms**. The language may retain them for poetic or dialectal variation, or the community may standardize on the most efficient (shortest) sequence for each concept.

### 9.2 Neologisms

When new concepts arise (e.g., "plastic", "diver"), speakers run the encoding algorithm online, producing a new word. If the word is adopted, it enters the lexicon.

---

## 10. Relationship to the Case System

The lexical system produces content words. These words are **inflected** for case using separate physical channels:

- **Nominative**: Mantle neutral, siphon centered
- **Accusative**: Mantle tilted left, siphon left
- **Dative**: Mantle tilted right, siphon right
- **Genitive**: Mantle raised, siphon up
- **Locative**: Mantle lowered, siphon down
- **Instrumental**: Mantle pulsed, siphon oscillating
- …

When signing a sentence, the signer produces a sequence of **case-marked words**:

$$
[\text{case}_1 + \text{word}_1] \; [\text{case}_2 + \text{word}_2] \; \dots
$$

Because case is marked externally, word order is free. The case markers disambiguate syntactic roles.

---

## 11. Summary

Let:

- $\mathcal{S} = \{s_1, \dots, s_n\}$: set of syllables, $n = 20$
- $\mathbf{V} = \{\mathbf{v}_1, \dots, \mathbf{v}_n\} \subset \mathbb{R}^d$: codebook of unit vectors
- $\lambda \in (0,1)$: decay factor (default 0.5)
- $\epsilon$: residual threshold for encoding
- $L_{\max}$: maximum word length (default 4)

**Word**: A finite sequence $\sigma = \langle s_{i_1}, s_{i_2}, \dots, s_{i_k}\rangle$ with $k \leq L_{\max}$.

**Meaning function**:

$$
\mu(\sigma) = \sum_{j=1}^k \lambda^{j-1} \mathbf{v}_{i_j}
$$

**Encoding function** (for target $\mathbf{t}$):

$$
\epsilon(\mathbf{t}) = \text{greedy\_pursuit}(\mathbf{t}, \mathbf{V}, \lambda, \epsilon, L{\max})
$$

**Properties**:

- $\mu \circ \epsilon$ approximates identity for concepts in the training distribution.
- The system is **compositional**: meaning of a word is the weighted sum of its parts.
- The system is **hierarchical**: earlier syllables dominate.
- The system is **learned**: the codebook $\mathbf{V}$ is optimized via differentiable training, and the semantic space is also learned from data.
