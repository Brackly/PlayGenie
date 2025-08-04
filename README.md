# PlayGenie

Generative model for intelligent playlist completion.

---

## 🧠 Problem Statement

### 🎯 The Goal

To develop a generative model that can **intelligently complete user playlists** by recommending songs that align with a user's musical preferences and context. The core objective is to **maximize the total engagement time** $T$ that a user spends on our music platform.

---

### 💡 Why does it matter? (Business value & impact)

Increasing user engagement time has several strategic benefits:

- A longer $T$ indicates **higher user satisfaction and retention**.
- As $T$ increases across the user base, it directly contributes to a **larger share of the music streaming market**.
- Higher engagement also offers **more monetization opportunities**, such as ads or subscriptions.

To achieve this, the model  must learn from incomplete playlists and user behavior to generate relevant song continuations that are personalized, coherent, and engaging.

---

## 🚧 Challenges & Motivations

1. **Expanding and Overwhelming Song Catalog**
    - **Discovery Gap**: Many songs that match a user’s taste or mood may remain undiscovered. Bridging this gap increases $T$.
    - **Choice Overload**: The volume of options causes decision fatigue. Intelligent completion reduces friction and guides users.

2. **Latent and Evolving User Preferences**
    - User intent and taste shift with time, mood, and context. Preferences are **latent** and hard to explicitly capture.

3. **Sequential and Contextual Complexity**
    - Playlist construction is sequential. Good continuation depends on tempo, mood, genre, and flow.

4. **Cold Start and Data Sparsity**
    - New users, rare songs, or short playlists provide minimal context, requiring strategies beyond popularity.

---

## 🧩 Modeling Approach

The  goal is to **maximize user engagement time $T$** by completing playlists that align with a user’s preferences, intent, and listening context.

This is formulated as a **conditional sequence-to-sequence generation problem**, where the model generates a continuation of a partial playlist conditioned on the user:

```math
f(\text{playlist}_{\leq n} \mid \mathbf{u}) \rightarrow \text{playlist}_{n+1:\,n+k}

```

## 🎼 Model Components

Where:

- $f$: the generative model
- $n$: the number of songs in the current playlist
- $k$: the number of songs to generate
- $\text{playlist}_{\leq n}$: the observed (partial) playlist — an ordered sequence of songs curated or consumed by the user
- $\mathbf{u}$: the **latent, contextual, and relational user representation**, incorporating:
  - Latent preferences (e.g., mood, tempo, genre)
  - Session-level context (e.g., time of day, recent interactions)
  - Graph-informed signals via message passing from user-item interaction graphs
- $\text{playlist}_{n+1: n+k}$: the predicted continuation — a sequence of $k$ songs meant to maximize coherence and engagement

---

## 📐 User Modeling via Inference

Let $D$ denote the user’s historical data — including previous playlists, sessions, and interactions. We treat $\mathbf{u}$ as a latent variable inferred from $D$ using Bayes' rule:

```math
P(\mathbf{u} \mid D) \propto P(D \mid \mathbf{u}) \cdot P(\mathbf{u})
```
This frames user modeling as a **latent variable inference problem**, where we infer $\mathbf{u}$ from historical observations. The generative model then conditions on $\mathbf{u}$ to predict the most engaging continuation:

```math
P(\text{playlist}_{n+1:\,n+k} \mid \text{playlist}_{\leq n}, \mathbf{u}) \cdot P(\mathbf{u})
```
By learning both terms:
* $P(\mathbf{u} \mid D)$
* $P(\text{playlist}\_{n+1:n+k} \mid \text{playlist}\_{\leq n}, \mathbf{u})$

enabling  **personalized and adaptive playlist generation**, even in sparse data regimes.

## 📉 Loss Function (ELBO)

**Variational Inference** is adopted to optimize the **Evidence Lower Bound (ELBO)** as the training objective.

Let:
- $\mathbf{u}$: latent user vector  
- $q(\mathbf{u} \mid \text{playlist}_{\leq n})$: encoder (inference network)  
- $p(\mathbf{u})$: prior distribution over latent states (e.g., standard Gaussian)  
- $p(\text{playlist}\_{n+1:n+k} \mid \text{playlist}\_{\leq n}, \mathbf{u})$: decoder (generative model)

The ELBO to maximize is:

```math
\mathcal{L}_{\text{VAE}} = 
- \mathbb{E}_{q(\mathbf{u} \mid \text{playlist}_{\leq n})} \left[
  \sum_{i=1}^{k} \log P(s_{n+i} \mid s_{\leq n+i-1}, \mathbf{u})
\right]
+ \beta \, D_{\text{KL}} \left[
  q(\mathbf{u} \mid \text{playlist}_{\leq n}) \,\|\, p(\mathbf{u})
\right]
```
This objective balances:
* **Reconstruction accuracy**: how well the predicted playlist matches the actual continuation
* **Regularization via KL divergence**: keeps the inferred posterior close to the prior

$\beta$ can be tuned to control the tradeoff between these two terms (as in $\beta$-VAE).
