## LANGUAGE MODELS ARE INJECTIVE AND HENCE INVERTIBLE

Giorgos Nikolaou ‡∗

Tommaso Mencattini † , ‡ , ∗

Donato Crisostomi †

Andrea Santilli †

Yannis Panagakis § , ¶

Emanuele Rodol` a †

† Sapienza University of Rome

‡ EPFL

§ University of Athens

¶ Archimedes RC

## ABSTRACT

Transformer components such as non-linear activations and normalization are inherently non-injective, suggesting that different inputs could map to the same output and prevent exact recovery of the input from a model's representations. In this paper, we challenge this view. First, we prove mathematically that transformer language models mapping discrete input sequences to their corresponding sequence of continuous representations are injective and therefore lossless, a property established at initialization and preserved during training. Second, we confirm this result empirically through billions of collision tests on six state-of-the-art language models, and observe no collisions. Third, we operationalize injectivity: we introduce SIPIT, the first algorithm that provably and efficiently reconstructs the exact input text from hidden activations, establishing linear-time guarantees and demonstrating exact invertibility in practice. Overall, our work establishes injectivity as a fundamental and exploitable property of language models, with direct implications for transparency, interpretability, and safe deployment.

## 1 INTRODUCTION

A core question in understanding large language models is whether their internal representations faithfully preserve the information in their inputs. Since Transformer architectures rely heavily on nonlinearities, normalization, and many-to-one attentions mechanisms, it is often assumed that they discard information: different inputs could collapse to the same hidden state, making exact recovery of the input impossible. This view motivates concerns around transparency, robustness, and safe deployment, as it suggests that the link between text and representation is inherently lossy .

<!-- image -->

⇒

Figure 1: The map from prompts to latent space is injective. SIPIT inverts it.

In this paper, we show that this intuition is misleading. Despite their apparent complexity, standard decoder-only Transformer language models (seen as maps from prompts to hidden states) are in fact almost-surely injective ; for essentially all parameter settings and during the course of training, different prompts yield different last-token representations.

Building upon this property, we further provide a practical algorithm, SIPIT, that reconstructs the exact input from hidden activations. To our knowledge, it is the first to guarantee exact recovery in provable linear time (worst case bound), often faster in practice, turning injectivity from a theoretical property into an operational tool.

Our approach. To establish our result, we take a rigorous mathematical view of Transformers as functions. The key idea is that their components (embeddings, LayerNorm, causal attention, MLPs, and residual wiring) are smooth and structured enough that the model, as a whole, behaves predictably with respect to its parameters. Using tools from real analysis, we show that collisions

∗ Equal contribution; author order settled via Mario Kart.

(two different prompts producing the exact same representation) can only occur on a set of parameter values that has measure zero; that is, they are mathematical exceptions rather than possibilities one should expect in practice. Moreover, we prove that common training procedures (gradient descent with standard step sizes) never move parameters into this exceptional set. In layman's terms, almost all models at initialization are injective, and training preserves this property.

Technically, our proofs rely on two ingredients. First, we establish that Transformers are realanalytic functions of their parameters, which allows us to reason precisely about when and where collisions could occur. Second, we construct parameter settings where no two prompts collide, and show that gradient descent (GD) does not collapse such separation, i.e., collisions remain a measurezero event. The end result is a finite-horizon guarantee : after any fixed number of training steps, and under mild assumptions, injectivity holds with probability one. We provide complete formal proofs of these statements.

Main result. Our central finding is that causal decoder-only Transformer language models are injective almost surely. Formally, consider one such model with embedding width d , at least one attention head per block, real-analytic components, finite vocabulary V , and finite context length K . Initialize its parameters θ at random, using any distribution that has a density 1 (such as Gaussian, uniform, or Xavier/Glorot), and train for any finite number T of GD steps with step sizes in (0 , 1) . Then, with probability one over the random initialization,

<!-- formula-not-decoded -->

̸

̸

i.e., the map from prompts s to last-token representations r (s ; θ T ) is injective across all prompts in V ≤ K . In short, collisions in practical settings form a measure-zero set, and neither initialization nor training will ever place a model inside that set.

Significance. Our result shows that in standard decoder-only Transformers, different prompts almost surely yield different last-token representations across all practically relevant parameter settings and training procedures. The guarantee is both generic (it fails only on a measure-zero set of pathological parameters) and practical (it holds at finite width, depth, and training time under common initializations).

Conceptually, we replace a long-assumed property with a rigorous theorem, showing that injectivity is not an asymptotic idealization but a structural consequence of the architecture itself. Technically, our analytic framework pinpoints when collisions can arise (through deliberate non-analytic choices such as quantization or tying), and clarifies that otherwise the model is inherently lossless. Importantly, it establishes that last-token states almost everywhere identify the input.

Finally, we turn this theoretical guarantee into an operational tool: our algorithm SIPIT uses gradient-based reconstruction to recover prompts exactly from internal activations, efficiently and with provable linear-time guarantees. This confirms empirically that collisions do not occur in practice. Beyond transparency and safety, this elevates invertibility to a first-class property of Transformer language models, enabling stronger interpretability, probing, and causal analyses.

## 2 TRANSFORMERS ARE INJECTIVE

Summary. In this section we show that decoder-only Transformers almost surely map different prompts to different hidden states. Collisions can only occur under measure-zero parameter choices, and gradient-based training never creates them. In simple terms, Transformer representations are structurally lossless.

Approach. We consider causal decoder-only Transformer language models with vocabulary V , finite context window K , and embedding dimension d . For an input sequence s ∈ V ≤ K , let r (s ; θ ) denote the final hidden representation at the last token position 2 , given parameters θ .

Our analysis relies on three facts:

1 Put simply, parameters are not drawn from a degenerate or hand-crafted set.

2 Wefocus on the last-token state, since it alone drives next-token prediction; earlier rows matter only insofar as they shape this final state. Injectivity at the last token is the property of real operational interest.

- (i) Real-analyticity. Each component of the architecture (embeddings, positional encodings, LayerNorm with ε &gt; 0 , causal attention, MLPs with analytic activations, residuals) is realanalytic in its parameters (see Appendix A.2 for the mathematical background). This smoothness implies that the set of parameter values causing two distinct prompts to collide is extremely thin (measure zero).
- (ii) Initialization. Standard initialization schemes (Gaussian, uniform, Xavier/Glorot, etc.) draw parameters from continuous distributions with densities, so they avoid measure-zero sets with probability one.
- (iii) Training. Gradient-based updates (including SGD and mini-batch/full-batch GD) preserve absolute continuity of the parameter distribution after any finite number of steps; thus, training cannot generate collisions.

These facts allow us to state and prove injectivity results without relying on asymptotics.

We begin by establishing the analytic structure of the architecture.

Theorem 2.1 (Transformers are real-analytic) . Fix embedding dimension d and context length K . Assume the MLP activation is real-analytic (e.g. tanh, GELU). Then for every input sequence s ∈ V ≤ K , the map

<!-- formula-not-decoded -->

is real-analytic jointly in the parameters θ and the input embeddings.

Sketch of proof (full proof in Appendix B, Proposition B.3). Each building block is real-analytic: polynomials (embeddings, projections), exponential and softmax (attention), reciprocal square root (LayerNorm with ε &gt; 0 ), analytic activations in the MLP, and affine maps. Real-analytic functions are closed under addition, multiplication, quotient, and composition. Since the Transformer is a finite composition of such blocks, the entire map is real-analytic.

This smoothness result drives everything that follows: it ensures that collisions, if they exist, are confined to measure-zero parameter sets. We now ask: what happens at initialization?

Theorem 2.2 (Almost-sure injectivity at initialization) . Let θ be drawn from any distribution with a density (e.g. Gaussian or uniform). Then for any two distinct prompts s , s ′ ∈ V ≤ K ,

<!-- formula-not-decoded -->

Sketch of proof (full proof in Appendix C, Theorem C.2). Fix s = s ′ and consider

<!-- formula-not-decoded -->

̸

By Theorem 2.1, h is real-analytic. A fundamental dichotomy of real-analytic functions states that either h is

<!-- image -->

-

Figure 2: Two real-analytic functions f 1 and f 2 and their difference f 1 -f 2 . Black contours show the zero sets, which form thin curves (measure zero) rather than regions of positive measure.

identically zero, or its zero set has Lebesgue measure zero (see Figure 2 for an illustration). Therefore, to rule out the pathological case h ≡ 0 it suffices to exhibit a single parameter setting where r (s ; θ ) = r (s ′ ; θ ) .

This can always be done: if s and s ′ differ at the last position (symbol or length), freeze the network so that the last state reduces to embedding plus position, and choose distinct rows; this already separates r (s) and r (s ′ ) . If instead they differ earlier, let i ⋆ be the first mismatch and set one attention head so the last position attends almost entirely to i ⋆ , encoding its token in the value; this forces different outputs for s and s ′ .

̸

Hence h is not identically zero, and so the collision set { θ : h ( θ ) = 0 } has Lebesgue measure zero. Since standard initializations have densities, the probability of sampling such θ is zero, and r (s ; θ ) = r (s ′ ; θ ) (injectivity) holds almost surely at initialization.

̸

According to Theorem 2.2, at initialization, collisions are mathematically impossible except on a vanishingly small set of parameter values. Finally, with the following Theorem we ensure training does not break injectivity.

Theorem 2.3 (Injectivity preserved under training) . Let θ 0 be initialized from a distribution with a density, and let θ T be the parameters after T steps of gradient descent with step sizes in (0 , 1) . Then with probability one,

<!-- formula-not-decoded -->

̸

̸

Sketch of proof (full proof in Theorems C.1 and C.5). At initialization, θ 0 is drawn from a distribution with a density, hence absolutely continuous. To break injectivity during training, GD would need to map this continuous law onto the measure-zero collision set identified in Theorem 2.2. We show this cannot happen.

Asingle GD step is the map ϕ ( θ ) = θ -η ∇L ( θ ) , where L is the training loss. Because the network and the softmax cross-entropy loss are real-analytic, ϕ is also real-analytic. Its Jacobian determinant det Dϕ ( θ ) is itself real-analytic and not identically zero (one can check this by evaluating at a simple parameter setting). Hence the set where det Dϕ = 0 has measure zero. Away from that set, the Inverse Function Theorem applies: ϕ is a smooth, locally invertible change of coordinates that can stretch or bend space but cannot collapse regions of positive volume onto lower-dimensional sets. Therefore, pushing forward an absolutely continuous distribution through ϕ yields another absolutely continuous distribution.

Since this argument holds for each step, any finite sequence of GD updates preserves absolute continuity of the parameter law. Combining with Theorem 2.2, which shows that collision sets are measure-zero, we conclude that r (s ; θ T ) = r (s ′ ; θ T ) almost surely for all s = s ′ .

̸

̸

Thus injectivity is not just an initialization property but remains true throughout training. A simple but important corollary follows.

Corollary 2.3.1 (SGD and mini-batch GD) . Under the assumptions of Theorem 2.3, the same conclusion holds when the updates are θ t +1 = θ t -η t ∇ θ L B t ( θ t ) with arbitrary (possibly random or adversarial) batch selections B t , thus including the singleton case of SGD and the full dataset.

Proof. The proof argument of Theorem 2.3 is unchanged: for each fixed batch B , the update map ϕ B ( θ ) = θ -η ∇L B ( θ ) is real-analytic with a Jacobian that is not identically zero. Indeed, the batch loss is the average L B = 1 |B| ∑ |B| i =1 L i , so at the point θ ⋆ from the single-sample proof (where the Jacobian determinant is sample-independent and nonzero) the batch Jacobian coincides with the single-sample one by linearity of differentiation, and its determinant is therefore also nonzero. Thus, the finite composition of such maps preserves absolute continuity of the parameter law.

Together with this robustness to different training regimes, we can also strengthen the guarantee itself: injectivity holds not just pairwise, but globally across finite sets of prompts.

Corollary 2.3.2 (Distinctness for finite sets) . For any finite set of prompts S ⊆ V ≤ K , the representations { r (s ; θ T ) : s ∈ S} are almost surely all distinct.

Proof.

<!-- formula-not-decoded -->

These results show that decoder-only Transformer language models are structurally injective: different prompts almost surely yield different last-token states. Collisions can be manufactured, e.g., through deliberate non-analytic choices (quantization, non-smooth activations), but in practical training pipelines, injectivity is guaranteed; extensive experiments in §4.1 confirm this empirically.

̸

Failure cases. We showed that non-injective transformers are overwhelmingly unlikely, though it is still possible for an adversary to construct collisions by hand. For instance, if two vocabulary items v i = v j are assigned exactly the same embedding vector, then any prompts differing only by swapping v i and v j yield identical representations. Likewise, if two absolute positional embeddings are made exactly equal and the remaining weights are tuned to suppress other positional signals,

Figure 3: Seeking collisions in a large-scale prompt set (§4.1). The minimum distances between last-token states are far above the collision threshold 10 -6 : (left) across layers for GPT-2 and Gemma-3 families (one dot per layer), (right) across depth for GPT-2 Small , where distances grow with depth.

<!-- image -->

one can force collisions between sequences that differ only at those positions. These scenarios, however, require deliberately engineered parameter choices: under continuous random initialization and standard training, the probability of such coincidences is zero.

## 3 EXACT PROMPT RECOVERY VIA SIPIT

In the previous section, we have proven that decoder-only Transformers are almost surely injective, i.e., different prompts map to different hidden states. We now show how this property can be used in practice to reconstruct the exact input prompt given hidden states at some layer. We call this algorithm SIPIT (Sequential Inverse Prompt via ITerative updates).

Formally, recall from §2 that the mapping from a prompt s to its last-token state is almost surely injective. Since the last state is itself a deterministic function of the hidden matrix at any layer ℓ , injectivity extends to the full representation

<!-- formula-not-decoded -->

We denote by h t (s) the row of H ( ℓ ) (s) at position t . In the following, the parameters θ and target layer ℓ are considered fixed and omitted for simplicity.

The algorithm exploits the causal structure of Transformers: the hidden state at position t depends only on the prefix ⟨ s 1 , . . . , s t -1 ⟩ and the current token s t . This means that if we already know the prefix, then the hidden state at position t uniquely identifies s t .

Example. Suppose the vocabulary is a, b, c and the true prompt is ⟨ a, b ⟩ . At t = 1 , the hidden state depends only on s 1 . By comparing the observed state with the three candidate states produced by trying a , b , and c , we can tell exactly which one matches, thus recovering s 1 = a . Then at t = 2 , we know the prefix ⟨ a ⟩ , so we try appending each candidate token and again match the resulting hidden state to recover s 2 = b . Iterating this procedure reconstructs the full sequence.

More generally, we can look at the 'one-step' map

<!-- formula-not-decoded -->

which gives the hidden state at step t for each possible next token, given the fixed prefix π = ⟨ s 1 , . . . , s t -1 ⟩ (here ⊕ denotes concatenation).

Remark. By the analytic arguments of §2, the one-step map is almost surely injective: with a fixed prefix, any two distinct tokens almost surely yield distinct hidden states.

This property makes sequence recovery straightforward. At each step t , given the hidden state ̂ h t and the already recovered prefix, we simply check which candidate token produces a matching hidden state. That token must be the true s t . Repeating this process recovers the entire sequence.

This leads to the SIPIT algorithm, shown in Algorithm 1. At every position, the algorithm cycles through vocabulary candidates (according to some policy such as random order or gradient-guided search) until it finds the unique match 3 , then appends it to the reconstructed prefix and moves on.

## Algorithm 1 SIP-IT: Sequential Inverse Prompt via Iterative Updates

```
Require: Observed layerℓ states ̂ H ( ℓ ) ∈ R T × d ; vocabulary V ; tolerance ε ≥ 0 . Ensure: Recovered sequence ̂ s = ⟨ ˆ s 1 , . . . , ˆ s T ⟩ . 1: ̂ s ←⟨⟩ 2: for t = 1 to T do 3: C ← ∅ ▷ tested candidates 4: for j = 1 to |V| do 5: v j ← POLICY ( V , C , ̂ s , ℓ ) ▷ new candidate token v j (see Alg. 2 and 3) 6: if ̂ h t ∈ A π,t ( v j ; ε ) then ▷ verify v j (see Def. D.2) 7: ̂ s ← ̂ s ⊕ v j ▷ hit! 8: break 9: else 10: C ← C ∪ { v j } 11: end if 12: end for 13: end for 14: return ̂ s
```

To rule out edge cases and analyze the computational cost of SIPIT, we now state a formal guarantee.

Theorem 3.1 (Correctness of SIPIT) . Under the assumptions of Theorem 2.3, given observed hidden states ̂ H ( ℓ ) , SIPIT recovers the true input sequence s with probability one in at most T |V| steps.

Sketch of proof (full proof in Appendix D, Thm. D.2, Prop. D.4). At each step, local injectivity ensures a unique token matches the observed state. As the policy spans the vocabulary, this token will be found in at most |V| trials. Induction over t = 1 , . . . , T completes the argument.

In short, SIPIT turns the almost-sure injectivity of Transformer representations into a constructive procedure: not only are hidden states unique identifiers of prompts, but the exact input sequence can be efficiently recovered in linear time, and often faster in practice. It is a structural property of Transformer representations, not a quirk of initialization or training.

## 4 EXPERIMENTS

Wepreviously proved that decoder-only Transformers are injective (§2) and introduced an algorithm, SIPIT, that leverages this property to recover the exact input prompt from hidden states at a given layer (§3). We now provide extensive empirical evidence supporting our theory by showing that distinct prompts yield distinct embeddings, i.e., no collisions occur by a large margin (§4.1). We then demonstrate that SIPIT successfully reconstructs the original input prompt (§4.2).

Environment. All experiments were run on a single NVIDIA A100 -SXM (64GB) GPU. Python 3.11, CUDA 12.2, PyTorch 2.8.0, and transformers 4.50.0 were used for all experiments. Reported runtimes refer to this setup.

## 4.1 SEARCHING FOR COLLISIONS

We collected 100k prompts by uniformly sampling from a mixture of four datasets: wikipedia-en 4 , C4 (Raffel et al., 2020), The Pile (Gao et al., 2020), and

3 In practice, we accept matches if the observed hidden state is within an ε -ball around the predicted one.

4 https://huggingface.co/datasets/wikimedia/wikipedia

Figure 4: Exhaustive collision search on the 10 closest prefix prompts. The boxplots look flat and uneventful, and that is the point: even under stress-test conditions with billions of candidate pairs, all minima stay well above the collision threshold, showing that nothing collapses.

<!-- image -->

python-github-code 5 . For each prompt, we extracted the last-token representation and systematically checked whether any two distinct prompts produced identical embeddings. This process required around 5 billion pairwise comparisons.

We observed no collisions across all models and layers: distinct prompts always yielded distinct last-token states. Figure 3 (left) shows the per-layer minimum distances for the Gemma3 pretrained (Team et al., 2025) and GPT-2 (Radford et al., 2019) families, with strictly positive values throughout. Table 1 complements this by reporting the same statistic for Llama-3.1-8B (Grattafiori et al., 2024), Mistral-7B-v0.1 (Jiang et al., 2023), Phi-4-mini-instruct (Microsoft et al., 2025) and TinyStories-33M (Eldan &amp; Li, 2023), again showing clear separation at the first, middle, and last layers.

Table 1: Minimum pairwise distance between last-token states in the first, middle, and final layers of four models. All values are well above the collision threshold 10 -6 (no collisions).

| Model           | L2 Distance (min)   | L2 Distance (min)   | L2 Distance (min)   |
|-----------------|---------------------|---------------------|---------------------|
|                 | layer 1             | layer L 2           | layer L             |
| Llama-3.1-8B    | 0.001               | 0.129               | 0.620               |
| Mistral-7B-v0.1 | 0.002               | 0.187               | 1.274               |
| Phi-4-mini-ins  | 0.014               | 1.336               | 9.020               |
| TinyStories-33M | 0.029               | 1.434               | 2.793               |

Finally, Figure 3 (right) zooms in on GPT-2 Small , revealing that these distances typically increase with depth. Additional results for GPT-2 Medium , GPT-2 Large and Gemma3 (1B, 4B, 12B) appear in Appendix E, confirming the same trend.

Figure 5: Sequence length vs. pairwise distance for GPT-2 . Min, mean, and max distances rise at short lengths and then stabilize, indicating consistent separability.

<!-- image -->

Figure 5 shows how pairwise distances between lasttoken states vary with prompt length in GPT-2 Small . Three patterns emerge: (i) the minimum distance is never close to zero at all lengths, and (ii) it grows rapidly at short lengths but then levels off, suggesting that beyond a moderate context size, adding tokens does not affect separability; (iii) the overall spread (min-max) stays bounded, with no sign of pathological collapses. Similar behavior is seen in Gemma3 (see Appendix E, Figure 9). Overall, clear margins emerge quickly and then stabilize, making collisions unlikely at any sequence length.

Exhaustive collision test. Different from previous experiments, in this setting (Figure 4), we restrict our analysis to the 10 prompts from the dataset mixture whose embeddings have the smallest last-token distances. For each of these prompts, we appended every vocabulary token and computed all pairwise distances between the resulting last-token states, effectively performing an exhaustive search over continuations and yielding more than 343 billion prompt pairs per model.

This exhaustive experiment helps rule out the possibility that earlier observations were simply due to chance in random sampling rather than a true absence of collisions. While a complete search over all possible prompts would be ideal, it is computationally infeasible. The number of unique prompts grows exponentially with sequence length, and the number of pairwise comparisons grows even faster. For context, even with single-token prompts and the vocabulary size of Gemma3-1B , there

5 https://huggingface.co/datasets/angie-chen55/python-github-code

are already over 34 trillion possible prompt pairs, making exhaustive evaluation entirely impractical. Our compromise still revealed structure: we identified 5 prompt pairs with highly similar last-token embeddings, suggesting overlapping semantic content and motivating us to ask whether distinct next tokens could preserve meaning, i.e., yield essentially identical last-token hidden states.

Figure 4 reports the resulting distributions (min/median/mean/max) as boxplots for both GPT-2 Small and Gemma3-1B , with distances far from zero (no collision), confirming local injectivity as predicted by our theory.

## 4.2 INVERTIBILITY RESULTS

We now test whether the theoretical injectivity translates into exact recovery on pre-trained models. Using SIPIT with only the hidden states at a fixed layer, we attempt to reconstruct the full prompt token-by-token for GPT-2 Small . We sample 100 prompts, with a 90% -10% split between meaningful sentences and random token sequences (to test robustness in unstructured cases), and attempt to reconstruct them from hidden states.

We compare against HARDPROMPTS (Wen et al., 2023), which leverages gradient signals for approximate

Table 2: Prompt inversion: SIPIT ensures exact recovery efficiently, unlike HARDPROMPTS (no recovery) or brute force (infeasible runtimes).

| Method       | Mean Time (s)        | Accuracy   |
|--------------|----------------------|------------|
| HARDPROMPTS  | 6132 . 59 ± 104 . 61 | 0.00       |
| BRUTEFORCE   | 3889 . 61 ± 691 . 17 | 1.00       |
| SIPIT (ours) | 28 . 01 ± 35 . 87    | 1 . 00     |

prompt discovery, and against a SIPIT ablation without the gradient-guided candidate policy (BRUTEFORCE).

Other inversion approaches (Morris et al., 2023a;b; Nazir et al., 2025) tackle a different setting altogether: they operate in black box access, using sequences of next-token logprobs or encoder logits rather than hidden states, and train auxiliary inverters to reconstruct text, at high computational cost. Their outputs are typically approximate and not guaranteed exact. These differences make them complementary but not directly comparable to our setting of training-free, exact inversion from hidden states in decoder-only LMs.

Figure 6: Inversion time as a function of depth. Runtimes rise only mildly across layers.

<!-- image -->

## 5 RELATED WORK

Our results connect to two active lines of research: theoretical analyses of Transformer architectures, and inverse problems in language modeling. We briefly review both to position our contributions.

Analytical properties of Transformers. Viewed as functions on R d , individual Transformer components are clearly non-injective: LayerNorm collapses along per-example statistics (Ba et al., 2016), residual connections can cancel, and in attention-only stacks, rank decays doublyexponentially with depth (Dong et al., 2021). Likewise, on the output side, the softmax bottleneck constrains the distributions reachable by language models (Yang et al., 2018). From this algebraic

Results are reported in Table 2. Across all prompts (20 tokens each), SIPIT recovers the exact sequence with 100% token-level accuracy (no errors, no collisions), matching the theoretical guarantee of linear-time convergence.

In contrast, HARDPROMPTS fails to recover the true input in most cases, while BRUTEFORCE eventually succeeds but at a prohibitive computational cost, requiring several orders of magnitude longer.

Finally, Figure 6 shows inversion times by layer for longer prompts (ranging from 20 to 200 tokens). Although deeper layers are costlier in principle (since verifying a candidate and computing gradients require traversing more blocks), the effect is minor: runtimes rise only slightly from first to last layer, and the scaling remains graceful overall. Likely, earlier layers need more iterations to converge, while deep layers store richer information that reduces the search effort. As a result, the net cost remains stable, confirming SIPIT is efficient across depth.

perspective, Transformers seem inherently many-to-one, while in a generative sense, they can also behave one-to-many when different prompts lead to the same continuation.

Our focus is different: we study the discrete-to-continuous map from prompts s ∈ V ≤ K to hidden states in R d . In this setting, analytic viewpoints on Transformer computation become powerful: treating each layer as a real-analytic map yields almost-sure guarantees that hold at finite width, depth, and training horizon. Recent work has adopted this angle for related properties: Jiang &amp; Haghtalab (2025) show that building blocks of modern architectures are almost always surjective , while Sutter et al. (2025) prove that Transformers at random initialization are almost surely injective with respect to the entire hidden-state matrix (and only at initialization).

Differently, we prove injectivity with respect to the parameters and at the task-relevant last-token state ; crucially, we show that injectivity is not an initialization artifact but persists under training .

Inverse problems in language modeling. Inverse problems seek to recover an unknown input x from observations y produced by a forward process y = f ( x ) (Sun et al., 2021). Within this landscape, language model inversion asks whether one can reconstruct a model's input prompt from outputs or internal signals.

Several approaches have explored this idea. Output-to-prompt methods infer prompts from generated continuations, yielding approximate reconstructions that are often semantically similar rather than exact (Zhang et al., 2024). Recent work by Morris and coauthors shows that model outputs are information-rich even in black-box settings: Morris et al. (2023b) train a separate inverter to map next-token probability vectors to text, and Nazir et al. (2025) extend this by taking sequences of logprobs, applying a linear compression to embedding dimension, and training an encoder-decoder inverter; this achieves higher exact-match rates but still without guarantees. Complementarily, Morris et al. (2023a) reconstruct text from encoder logits via a trained iterative inverter. These contributions highlight privacy risks when probabilities or embeddings are exposed, but they differ from our setting: they rely on trained inverters, remain approximate, and do not invert hidden states of decoder-only LMs.

A related line of work frames the task as automated prompt optimization, casting prompt design as discrete sequence optimization aligned with downstream performance (Guo et al., 2025; Sun et al., 2022; Deng et al., 2022); methods such as AutoPrompt (Shin et al., 2020) and Hard Prompts Made Easy (Wen et al., 2023) use gradient signals to discover effective, but approximate, prompts.

Unlike prior work, which yields approximate reconstructions from outputs, logits, or logprobs, our approach is training-free, efficient, and comes with provable linear-time guarantees for exact recovery from internal states.

## 6 DISCUSSION AND CONCLUSIONS

This work establishes that decoder-only Transformers are almost surely injective: distinct prompts produce distinct hidden states under standard initialization and training. Building on this structural result, we introduced SIPIT, the first algorithm that can recover the exact input sequence from hidden activations, with provable linear-time guarantees. Together, these contributions move injectivity from an informal belief to a rigorously grounded and operational property of language models.

The scientific impact is clear. Our findings reconcile two competing views in the community: Transformers as 'lossy' due to nonlinearities, normalization, and many-to-one attention, versus language models as injective in their hidden representations. We advocate viewing language models as maps on the sequence space rather than the embedding space; under this perspective, we prove that all information about the input sequence is almost surely preserved end-to-end. The constructive inversion offered by SIPIT strengthens this point in practice, establishing a clean baseline for interpretability and auditing: if probes or inversion methods fail, it is not because the information is missing. For mechanistic interpretability in particular, injectivity guarantees that last-token states faithfully encode the full input, giving a sound foundation for causal and probing analyses.

Beyond theory, the findings carry practical and legal implications. Hidden states are not abstractions but the prompt in disguise. Any system that stores or transmits them is effectively handling user text itself. This affects privacy, deletion, and compliance: even after prompt deletion, embeddings

retain the content. Regulators have sometimes argued otherwise; for example, the Hamburg Data Protection Commissioner claimed that weights do not qualify as personal data since training examples cannot be trivially reconstructed (HmbBfDI, 2024). Our results show that at inference time user inputs remain fully recoverable. There is no 'free privacy' once data enters a Transformer.

Finally, this work opens several directions. Extending the analysis to multimodal architectures such as music and vision Transformers is an open problem. Studying approximate inversion under noise or quantization will clarify how robust invertibility remains in practice. Bridging these technical insights with evolving regulatory frameworks will be crucial for safe and responsible deployment.

## REPRODUCIBILITY STATEMENT

We provide complete resources to ensure reproducibility of our results. The assumptions, definitions, and full proofs can be found in section 2 and sections A to C (analytic tools and model specification in sections A and B; almost-sure injectivity and preservation under training in section C; SIP-IT correctness, verifier, and margin analysis in section D). Implementation details for SIP-IT, including pseudocode, are provided in section 3 and algorithm 1 and further elaborated in section E. Our experimental setup (hardware and software versions) is described in section 4, while dataset details and the prompt-sampling procedure for the 100k-prompt benchmark are given in section 4.1. Finally, the supplementary materials include an anonymized code repository with end-to-end scripts, fixed seeds, configuration files, and a comprehensive README with step-by-step reproduction instructions.

## REFERENCES

- W. E. Aitken. General topology. part 4: Metric spaces, 2020. URL https://public.csusm. edu/aitken\_html/Essays/Topology/metric\_spaces.pdf . 22
2. Shane Arora, Hazel Browne, and Daniel Daners. An alternative approach to fr´ echet derivatives. Journal of the Australian Mathematical Society , 111(2):202-220, 2021. 22
3. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450 , 2016. URL https://arxiv.org/abs/1607.06450 . 8
4. Jos´ e E Chac´ on and Tarn Duong. Higher order differential analysis with vectorized derivatives. arXiv preprint arXiv:2011.01833 , 2020. 17
5. Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric P. Xing, and Zhiting Hu. Rlprompt: Optimizing discrete text prompts with reinforcement learning, 2022. URL https://arxiv.org/abs/2205.12548 . 9
6. Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In Proceedings of the 38th International Conference on Machine Learning (ICML) , volume 139 of Proceedings of Machine Learning Research , 2021. URL https://proceedings.mlr.press/v139/dong21a.html . 8
7. Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english?, 2023. URL https://arxiv.org/abs/2305.07759 . 7
8. Gerald B Folland. Real analysis . Pure and Applied Mathematics: A Wiley Series of Texts, Monographs and Tracts. John Wiley &amp; Sons, Nashville, TN, 2 edition, March 1999. 23, 37
9. Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling, 2020. URL https://arxiv.org/ abs/2101.00027 . 6
10. Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava

Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzm´ an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur C ¸ elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ ıtor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Jun-

jie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783 . 7

- Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yujiu Yang. Evoprompt: Connecting llms with evolutionary algorithms yields powerful prompt optimizers, 2025. URL https://arxiv.org/abs/2309.08532 . 9
- Harold V Henderson and Shayle R Searle. The vec-permutation matrix, the vec operator and kronecker products: A review. Linear and multilinear algebra , 9(4):271-288, 1981. 17, 34
- HmbBfDI. Discussion paper: Large language models and personal data, 2024. URL https://datenschutz-hamburg.de/fileadmin/user\_upload/HmbBfDI/ Datenschutz/Informationen/240715\_Discussion\_Paper\_Hamburg\_DPA\_ KI\_Models.pdf . 10
- Roger A. Horn and Charles R. Johnson. Matrix Analysis . Cambridge University Press, Cambridge, 2 edition, 2013. 33
- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´ elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ ee Lacroix, and William El Sayed. Mistral 7b, 2023. URL https: //arxiv.org/abs/2310.06825 . 7
- Haozhe Jiang and Nika Haghtalab. On surjectivity of neural networks: Can you elicit any behavior from your model? arXiv preprint arXiv:2508.19445 , 2025. URL https://arxiv.org/ abs/2508.19445 . 9
- Tamara G. Kolda and Brett W. Bader. Tensor decompositions and applications. SIAM Review , 51(3):455-500, 2009. doi: 10.1137/07070111X. URL https://doi.org/10.1137/ 07070111X . 15

- Steven G Krantz and Harold R Parks. A primer of real analytic functions . Springer Science &amp; Business Media, 2002. 20
- Andrew D. Lewis. Chapter 1: Holomorphic and real analytic calculus. Notes on Global Analysis, Vol. 1, Queen's University, February 2014. URL https://mast.queensu.ca/˜andrew/ teaching/math942/pdf/1chapter1.pdf . Version: 2014-02-28. 16, 17, 36
- David G. Luenberger. Optimization by vector space methods . Wiley-Interscience, 1997. 22
- Jan R. Magnus and Heinz Neudecker. Matrix differential calculus with applications in statistics and Econometrics . John Wiley &amp; Sons, Inc, 2019. 22, 32
- Microsoft, :, Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Arindam Mitra, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, and Xiren Zhou. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras, 2025. URL https://arxiv.org/abs/2503.01743 . 7
- Boris Mityagin. The zero set of a real analytic function. arXiv preprint arXiv:1512.07276 , 2015. 17
- John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexander M. Rush. Text embeddings reveal (almost) as much as text, 2023a. URL https://arxiv.org/abs/2310.06816 . 8, 9
- John X. Morris, Wenting Zhao, Justin T. Chiu, Vitaly Shmatikov, and Alexander M. Rush. Language model inversion, 2023b. URL https://arxiv.org/abs/2311.13647 . 8, 9
- James R. Munkres. Topology . Prentice Hall, Upper Saddle River, NJ, 2 edition, 2000. 22, 23
- Murtaza Nazir, Matthew Finlayson, John X. Morris, Xiang Ren, and Swabha Swayamdipta. Better language model inversion by compactly representing next-token distributions, 2025. URL https://arxiv.org/abs/2506.17090 . 8, 9
- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. URL https://api. semanticscholar.org/CorpusID:160025533 . 7
- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-totext transformer. Journal of Machine Learning Research , 21(140):1-67, 2020. URL http: //jmlr.org/papers/v21/20-074.html . 6
- Walter Rudin. Principles of Mathematical Analysis . McGraw-Hill, New York, 3 edition, 1976. 23, 37
- Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts, 2020. URL https://arxiv.org/abs/2010.15980 . 9
- Michael Spivak. Calculus on manifolds . Westview Press, Philadelphia, PA, January 1971. 23
- Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, and Xipeng Qiu. Black-box tuning for language-model-as-a-service, 2022. URL https://arxiv.org/abs/2201.03514 . 9

- Zhaodong Sun, Fabian Latorre, Thomas Sanchez, and Volkan Cevher. A plug-and-play deep image prior. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pp. 8103-8107. IEEE, June 2021. doi: 10.1109/icassp39728.2021. 9414879. URL http://dx.doi.org/10.1109/ICASSP39728.2021.9414879 . 9
- Denis Sutter, Julian Minder, Thomas Hofmann, and Tiago Pimentel. The non-linear representation dilemma: Is causal abstraction enough for mechanistic interpretability?, 2025. URL https: //arxiv.org/abs/2507.08802 . 9, 31
- Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´ e, Morgane Rivi` ere, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Ga¨ el Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, Andr´ as Gy¨ orgy, Andr´ e Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Pluci´ nska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim P˜ oder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and L´ eonard Hussenot. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503.19786 . 7
- Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery, 2023. URL https://arxiv.org/abs/2302.03668 . 8, 9, 45
- Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. Breaking the softmax bottleneck: A high-rank rnn language model. In International Conference on Learning Representations (ICLR) , 2018. URL https://arxiv.org/abs/1711.03953 . 8
- Collin Zhang, John X. Morris, and Vitaly Shmatikov. Extracting prompts by inverting llm outputs, 2024. URL https://arxiv.org/abs/2405.15012 . 9

## A PRELIMINARIES

This section fixes notation the notation used throughout the main paper and the appendix (subsection A.1), and it introduces real-analyticity as the organizing theme (subsection A.2). We first review the vector-space notion and its basic closure/composition properties (subsubsection A.2.1), together with a zero-set principle used in measure-zero arguments. We then extend these ideas to maps between matrix spaces (subsubsection A.2.2) via vectorization/matricization and note that analyticity is preserved under matrix compositions. To streamline later proofs, we summarize real-analytic building blocks commonly used in transformer layers-polynomials, exponential/logarithm, softmax, row normalization, matrix products, Hadamard scaling, and stacking (subsubsection A.2.3). Finally, in subsection A.3, we collect differential and topological tools-Fr´ echet derivatives and the Hessian, standard facts on R p , the inverse function theorem, and pushforwards/absolute continuity-which we use for local invertibility and absolute-continuity arguments. Readers already comfortable with these topics can skim now and return to specific subsections as needed.

## A.1 NOTATION

For arbitrary T ∈ N , we write [ T ] = { 1 , 2 , . . . , T } to denote the set of positive integers up to T . Additionally, we denote the strictly positive real numbers as R + = (0 , ∞ ) and the non-negative real numbers as R + 0 = [0 , ∞ ) . Similarly, we let N 0 = N ∪ { 0 } .

Discrete sets are denoted by uppercase calligraphic letters V , and a sequence of length K is denoted by lowercase letters: s = ⟨ s 1 , . . . , s K ⟩ ∈ V K . We write | s | = K to denote the length of the sequence. The set of non-empty sequences of length at most K is denoted as V ≤ K = ⋃ K k =1 V k . Non-discrete sets are denoted by uppercase calligraphic bold-face letters B .

Remark 1. We will often refer to a discrete set V as the vocabulary and to an element s ∈ V ≤ K as an input, context, or prompt.

Matrices (vectors) are denoted by uppercase (lowercase) bold-face letters: X ∈ R d 1 × d 2 ( x ∈ R d ). For vectors and matrices, we frequently use standard norms and common matrix operations. The Hadamard and Kronecker products are defined following Kolda &amp; Bader (2009):

- p -norm: For a vector x ∈ R d , the ℓ p norm is defined as

<!-- formula-not-decoded -->

- Frobenius norm: For a matrix X ∈ R d 1 × d 2 , the Frobenius norm is defined as

<!-- formula-not-decoded -->

- Hadamard product: The Hadamard (element-wise) product is defined for vectors and matrices of the same shape:

<!-- formula-not-decoded -->

where x , y ∈ R d and X , Y ∈ R d 1 × d 2 .

- Kronecker product: The Kronecker product of X ∈ R d 1 × d 2 and Z ∈ R d 3 × d 4 is denoted X ⊗ Z and defined blockwise as

<!-- formula-not-decoded -->

We denote the all-zeros matrix of size m × n as 0 m × n , and the all-zeros vector of length m as 0 m . Similarly, we write 1 m for the all-ones vector of length m , and I m (or I m × m when dimensions must be explicit) for the m × m identity matrix.

Let f : V ≤ K × R p → R d be a function over a finite vocabulary V and K ∈ N . We refer to f as the model , to its first argument as the input sequence , and to its second argument as the parameters .

Remark 2. Throughout our analysis, we assume a finite set of possible input sequences, reflecting the practical limitations and design choices of modern LLMs, specifically the bounded context length.

Remark 3. We take the codomain of the model to be R d , corresponding to the space of token embeddings. This allows us to study how the final embedding (typically used to compute next-token probabilities) depends on both the input sequence and the model parameters.

## A.2 REAL-ANALYTICITY

We now introduce the central notion for our analysis: real-analyticity. In its standard form, realanalyticity is defined for functions f : U → R n , where U ⊆ R m is an open set. Since the transformer architecture is naturally expressed in terms of matrices, it will be convenient to extend this notion to maps of the form f : R m × n → R a × b .

Multi-index notation. We use multi-index notation for both vectors and matrices.

Vector case. Let α = ( α 1 , . . . , α m ) ⊤ ∈ N m 0 and x , y ∈ R m . Define:

<!-- formula-not-decoded -->

Matrix case. Let A = ( α uv ) ∈ N m × n 0 and X , Y ∈ R m × n . Define:

<!-- formula-not-decoded -->

Given an open set U ⊆ R m and a map f : U → R , we write

<!-- formula-not-decoded -->

for the mixed partial derivative (when it exists). Unless stated otherwise, we assume f ∈ C ∞ ( U ) , so d α f exists and is continuous for all α ∈ N m 0 ; for vector-valued maps f = ( f 1 , . . . , f n ) the operator d α acts componentwise. We also use the convention d 0 f = f .

## A.2.1 REAL-ANALYTIC FUNCTIONS WITH VECTOR INPUTS

Definition A.1 (Real-analytic functions, Lewis 2014, Definition 1.1.3) . Let U ⊆ R m be open. A function f : U → R is real-analytic on U if, for every y ∈ U , there exist coefficients { c α ∈ R } α ∈ N m 0 and r &gt; 0 such that

<!-- formula-not-decoded -->

for all x ∈ U with ∥ x -y ∥ 2 &lt; r . The set of real-analytic functions on U is denoted by C ω ( U ) .

A map f : U → R n is real-analytic on U if each of its components f 1 , . . . , f n : U → R is real-analytic. The set of such maps is denoted C ω ( U ; R n ) .

Remark 4. To establish real-analyticity of a vector-valued mapping (e.g., an MLP , attention mechanism, or LayerNorm), it suffices to prove real-analyticity of each scalar component.

Proposition A.1 (Closure properties, Lewis 2014, Proposition 1.2.1) . Let f, g : R m → R be realanalytic maps. Then, the following hold:

1. Addition: f + g ∈ C ω ( R m ) .

2. Product: fg ∈ C ω ( R m ) .

Proposition A.2 (Composition, Lewis 2014, Proposition 1.2.2) . Let f : R m → R n and g : R n → R k be real-analytic maps. Then, the composition g ◦ f : R m → R k is real-analytic.

3. Quotient: If g ( x ) = 0 for all x ∈ R m , then f/g ∈ C ω ( R m ) .

̸

Remark 5. For simplicity, we do not state the closure properties in their most general form, where f and g may be defined on different open subsets of R m . This avoids additional notation involving intersections of domains. Since every function of interest in our later analysis is defined on the whole space R m , this restriction entails no loss of generality.

Theorem A.1 (Zero sets of nontrivial real-analytic maps Mityagin 2015) . Let U ⊆ R m be connected and open, and let f ∈ C ω ( U ; R n ) . If f ̸≡ 0 n , then its zero set has Lebesgue measure zero in R m (i.e. Leb m ( Z ( f ) ) = 0 ). Equivalently, if there exists x ∈ U with f ( x ) = 0 n , then Leb m ( f -1 ( { 0 n } ) ) = 0 .

<!-- formula-not-decoded -->

Remark 6. The result in Mityagin (2015) is stated for scalar-valued maps f : U → R . The extension to vector-valued maps f = ( f 1 , . . . , f n ) : U → R n is immediate: the zero set of f is the intersection of the zero sets of its scalar components,

<!-- formula-not-decoded -->

and if f ̸≡ 0 n , then at least one component f j ̸≡ 0 , so Z ( f ) ⊆ Z ( f j ) , which has measure zero by the scalar case.

## A.2.2 REAL-ANALYTIC FUNCTIONS WITH MATRIX INPUTS

Definition A.2 (Real-analyticity on matrix spaces) . Let U ⊆ R m × n be open. A function f : U → R is real-analytic on U if, for every Y ∈ U , there exist coefficients { c A ∈ R } A ∈ N m × n 0 and r &gt; 0 such that

<!-- formula-not-decoded -->

for all X ∈ U with ∥ X -Y ∥ F &lt; r .

A map f : U → R a × b is real-analytic on U if each of its components f ij : U → R is real-analytic. The set of such maps is denoted C ω ( U ; R a × b ) .

Remark 7. In the special case where n = b = 1 , the domain and codomain reduce to R m and R a , respectively. Then Definition A.2 recovers Definition A.1. Thus, Definition A.2 generalizes real-analyticity to functions between matrix spaces.

Definition A.3 (Vectorization and matricization Operators) . Let vec m,n : R m × n → R mn denote the standard vectorization operator , which stacks the columns of a matrix into a single column vector (Henderson &amp; Searle, 1981).

We also define the corresponding matricization operator mat m,n : R mn → R m × n . As shown in Chac´ on &amp; Duong 2020, the vectorization and matricization operators are mutual inverses:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, if x ∈ R mn and X ∈ R m × n are related by vectorization and matricization, i.e., x = vec m,n ( X ) and X = mat m,n ( x ) , then their norms coincide:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Definition A.4 (Vectorized Form of Function) . Let U ⊆ R m × n be open and ˜ U = vec m,n ( U ) (also open since vec is a linear homeomorphism). We denote the vectorized form of a function f : U → R a × b as

̸

Equivalently, for all X ∈ U :

<!-- formula-not-decoded -->

Lemma A.1 (Equivalence real-analyticity) . Let U ⊆ R m × n be open, ˜ U = vec m,n ( U ) , and let f : U → R a × b with its vectorized form ˜ f : ˜ U → R ab .

1. f is real-analytic at Y (in the sense of Definition A.2).

Fix Y ∈ U and set y = vec m,n ( Y ) ∈ ˜ U . Then the following are equivalent:

2. ˜ f is real-analytic at y (in the sense of Definition A.1).

Proof. We begin by establishing the correspondence between matrix and vector indices in R k × ℓ and R kℓ . For s ∈ [ kℓ ] , define:

<!-- formula-not-decoded -->

Then ( u ( s ) , v ( s )) ∈ [ k ] × [ ℓ ] gives the matrix coordinates corresponding to the s th entry of the vectorization. Conversely, for ( u, v ) ∈ [ k ] × [ ℓ ] , define:

to recover the linear index.

<!-- formula-not-decoded -->

When clear from context, we omit arguments and simply write u , v , or s for readability.

Let X , Y ∈ R m × n , with vectorizations x = vec m,n ( X ) and y = vec m,n ( Y ) . For a vector multiindex α ∈ N mn 0 , define the corresponding matrix multi-index A α := mat m,n ( α ) , so that:

<!-- formula-not-decoded -->

Similarly, for a matrix multi-index A ∈ N m × n 0 , define the corresponding vector multi-index α A := vec m,n ( A ) , giving:

<!-- formula-not-decoded -->

Now let M ∈ U , and let m = vec m,n ( M ) ∈ ˜ U . By definition of the vectorization,

<!-- formula-not-decoded -->

This coordinate-wise correspondence underlies the equivalence stated in the lemma.

( ⇒ ) Assume f is real-analytic at Y . Then by Definition A.2, there exists r &gt; 0 and, for each ( u, v ) , coefficients { c ( uv ) A } A ∈ N m × n 0 such that:

<!-- formula-not-decoded -->

Using Equation 11, each component ˜ f s of ˜ f can be expressed as:

<!-- formula-not-decoded -->

This series converges for all x ∈ ˜ U with ∥ x -y ∥ 2 = ∥ X -Y ∥ F &lt; r . Hence, each scalar component of ˜ f has a convergent power series at y , proving that ˜ f is real-analytic there.

( ⇐ ) The reverse direction follows by symmetry: assume ˜ f is real-analytic at y , write the expansion at y using definition Definition A.1, and repeat the argument using Equation 10 to construct component-wise expansions for f uv at Y .

Remark 8. Consider the function f = vec m,n : R m × n → R mn × 1 , which vectorizes an m × n matrix by stacking its columns. Its corresponding vectorized form is

<!-- formula-not-decoded -->

since x ∈ R mn is already a column vector . This composition yields the identity map on R mn , which is clearly real analytic. Therefore, by Lemma A.1, both vec m,n is real analytic, and similarly, so is mat m,n . It is now evident that the composition of two matrix-valued real-analytic function is real-analytic, and we will prove it.

Proposition A.3 (Composition on matrix spaces is real-analytic) . Suppose f : R m × n → R a × b and g : R a × b → R p × q are real-analytic (in the sense of Definition A.2). Then g ◦ f : R m × n → R p × q is real-analytic.

Proof. Consider the vectorized forms

<!-- formula-not-decoded -->

By Lemma A.1, f is real-analytic iff ˜ f is, and g is real-analytic iff ˜ g is. Hence ˜ f and ˜ g are realanalytic maps between Euclidean spaces.

The vectorized form of the composition is

<!-- formula-not-decoded -->

where we inserted the identity (mat a,b ◦ vec a,b )( X ) = X . By the vector-space composition property (Proposition A.2), ˜ g ◦ ˜ f is real-analytic on R mn . Applying Lemma A.1 once more, we get that g ◦ f is real-analytic.

## A.2.3 REAL ANALYTICITY OF COMMON COMPONENTS

We now collect several building blocks that will be used repeatedly. Throughout, all maps are defined on R m × n , an open set, so Definition A.2 applies.

Proposition A.4 (Polynomials are real-analytic) . Let p : R m → R be a polynomial in the coordinates of x ∈ R m , i.e., p ( x ) = ∑ | α |≤ d a α x α for some d ∈ N 0 and coefficients a α ∈ R . Then p ∈ C ω ( R m ) .

Proof. Polynomials are C ∞ , and d α p ≡ 0 whenever | α | &gt; d . Hence the Taylor expansion of p at any y ∈ R m truncates:

<!-- formula-not-decoded -->

which holds for all x ∈ R m (radius r = + ∞ ). Therefore p is real-analytic.

Proposition A.5 (The exponential is real-analytic) . The map exp : R → (0 , ∞ ) is real-analytic on R .

Proof. Define E ( x ) := ∑ ∞ k =0 x k k ! . By the ratio test this power series has infinite radius of convergence, hence converges absolutely for all x ∈ R . Standard results on power series imply that E is C ∞ on R and can be differentiated termwise within its radius of convergence; in particular, for every j ∈ N 0 ,

<!-- formula-not-decoded -->

Fix y ∈ R . Taylor's theorem for power series then yields

<!-- formula-not-decoded -->

which is a convergent power series in x -y with infinite radius of convergence. Hence E is realanalytic at every y ∈ R . As E is the usual exponential function defined by its power series, exp is real-analytic on R .

Proposition A.6 (The logarithm is real-analytic) . The map log : (0 , ∞ ) → R is real-analytic on (0 , ∞ ) .

Proof. For brevity, we present only a proof sketch;

The exponential map exp : R → (0 , ∞ ) is real-analytic with exp ′ ( y ) = 0 for all y . By the realanalytic inverse function theorem (see Krantz &amp; Parks 2002, Thm. 2.3.1), its local inverse log is real-analytic on (0 , ∞ ) .

̸

Proposition A.7 (Softmax is real-analytic) . The map softmax : R m → R m with components is real-analytic on R m .

Proof. Fix i . The numerator x ↦→ e x i is the composition of the coordinate projection π i ( x ) = x i (a linear, hence real-analytic, map) with exp ; by Proposition A.5 and the composition rule in Proposition A.1, it is real-analytic. The denominator

<!-- formula-not-decoded -->

is a finite sum of real-analytic functions, hence real-analytic. Moreover, H ( x ) &gt; 0 for all x ∈ R m because e x j &gt; 0 . Therefore, by the quotient rule in Proposition A.1, the map

<!-- formula-not-decoded -->

is real-analytic on R m . Since this holds for each i = 1 , . . . , m , the vector-valued map softmax is real-analytic.

Proposition A.8 (Row normalization is real-analytic on positive row-sum domain) . Let

<!-- formula-not-decoded -->

Define RN( Y ) = diag( Y1 T ) -1 Y on D T . Then RN : D T → R T × T is real-analytic (in the sense of Definition A.2).

Proof. The map Y ↦→ s := Y1 T is linear, hence real-analytic. On (0 , ∞ ) T , the entrywise reciprocal s ↦→ s ⊙ ( -1) is real-analytic (componentwise t ↦→ 1 /t ). The map s ↦→ diag( s ) is linear. Matrix multiplication ( A , Y ) ↦→ AY is real-analytic (Proposition A.10). Composing these gives RN( Y ) = diag( Y1 T ) -1 Y real-analytic on the open set D T .

Proposition A.9 (Entrywise matrix polynomials are real-analytic) . Fix m,n ∈ N . For coefficients { c A ∈ R } A ∈ N m × n 0 and some d ∈ N 0 , define the function p : R m × n → R by:

<!-- formula-not-decoded -->

where X A = ∏ m u =1 ∏ n v =1 X A uv uv as defined in the multi-index notation above. Then p is realanalytic on R m × n (in the sense of Definition A.2).

Moreover, if f : R m × n → R a × b has component functions f ij of the form Equation 13, then f is real-analytic.

Proof. Consider the vectorized form ˜ p := p ◦ mat m,n : R mn → R . Using the coordinate identification from equation 11-equation 10, each monomial satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α A = vec m,n ( A ) . Hence:

is real-analytic.

Proof. Each scalar component of g (respectively h ) is exactly one scalar component of some f ℓ , hence real-analytic. Therefore g and h are real-analytic by definition Definition A.2.

Proposition A.13 (Noncommutative matrix polynomials are real-analytic) . Let n, p, q ∈ N , let X ∈ R n × n , and fix coefficient matrices A k ∈ R p × n and B k ∈ R n × q for k = 0 , . . . , d . Define

<!-- formula-not-decoded -->

Then f is real analytic in the sense of Definition A.2.

Proof. The identity map X ↦→ X is linear, hence a degree1 entrywise polynomial; by Proposition A.9 it is real-analytic. Assume X ↦→ X k is real-analytic. With f ( X ) = X k and g ( X ) = X , Proposition A.10 yields X k +1 = f ( X ) g ( X ) real-analytic; by induction, all powers X ↦→ X k are real-analytic.

For each k , left/right multiplication by fixed matrices preserves real-analyticity via Proposition A.10: since the constant maps X ↦→ A k and X ↦→ B k are real-analytic (components are constant polynomials), the composition X ↦→ A k X k B k is real-analytic. Finally, f is a finite sum of real-analytic maps, hence real-analytic by closure under addition (apply Proposition A.1 componentwise).

Remark 9. We highlight several standard constructions that yield real-analytic maps, omitting proofs for brevity:

<!-- formula-not-decoded -->

which is a standard multivariate polynomial in x ∈ R mn . By Proposition A.4, such functions are real-analytic on all of R mn , so ˜ p ∈ C ω ( R mn ) . By Lemma A.1, this implies p is real-analytic on R m × n .

For the second claim, observe that if each f ij is a scalar polynomial of the form Equation 13, then each f ij is real-analytic by the argument above. Hence, by Definition A.2, f is real analytic.

Proposition A.10 (Matrix product of real-analytic factors) . Let the functions f : R m × n → R p × r and g : R m × n → R r × q be real-analytic. Then, h : R m × n → R p × q defined as h ( X ) = f ( X ) g ( X ) , is real-analytic on R m × n .

<!-- formula-not-decoded -->

Each factor f ik and g kj is a real-analytic scalar map by assumption; their product is real-analytic by Proposition A.1, and a finite sum of real-analytic functions is real-analytic. Thus every h ij is real-analytic, hence h is real-analytic.

Proposition A.11 (Hadamard (element-wise) scaling) . Let A ∈ R m × n be a fixed matrix. Then, the map f : R m × n → R m × n defined as f ( X ) = A ⊙ X is real-analytic on R m × n .

Proof. Componentwise, ( A ⊙ X ) ij = A ij X ij is a product of a constant and a coordinate function, hence a polynomial (degree ≤ 1 ) and thus real-analytic.

Proposition A.12 (Concatenation/stacking of real-analytic blocks) . Let f ℓ : R m × n → R p × q ℓ be real-analytic for ℓ ∈ [ L ] . The horizontal concatenation operation g : R m × n → R p × ( q 1 + ··· + q L ) , defined as:

<!-- formula-not-decoded -->

is real-analytic. Likewise, if f ℓ : R m × n → R p ℓ × q are real-analytic, then the vertical stacking operation h : R m × n → R ( p 1 + ··· + p L ) × q , defined as:

<!-- formula-not-decoded -->

- Affine and bilinear maps. Functions of the form X ↦→ AXB + C are real-analytic, as they are obtained via matrix multiplication and addition of constant matrices (Proposition A.10, Proposition A.1).
- Algebraic expressions in X . Any expression constructed from X using finitely many additions and matrix multiplications with fixed coefficient matrices, e.g. A 0 + A 1 XB 1 + A 2 XB 2 XC 2 - defines a real-analytic map. This follows from repeated application of Proposition A.10 and closure under addition.
- Scalar polynomial invariants. Coordinate functions X ij , the trace tr( X ) , all principal and nonprincipal minors, and the determinant det( X ) are scalar polynomials in the entries of X , and hence real-analytic by Proposition A.9.

## A.3 DIFFERENTIAL, MEASURE-THEORETIC, AND TOPOLOGICAL TOOLS

This subsection collects the minimal calculus, measure, and topology we will use later. In finite dimensions, Fr´ echet derivatives let us speak uniformly about Jacobians and Hessians; basic Euclidean topology lets us control neighborhoods and compactness; the inverse function theorem gives local invertibility; and pushforwards/absolute continuity formalize how distributions transform under measurable maps.

Definition A.5 (Fr´ echet derivative (Luenberger, 1997, §7.2-§7.3)) . Let U ⊆ R m open, and consider a function f : U → R n . We say that f is Fr´ echet differentiable at x ∈ U if there exists a bounded linear map A : R m → R n such that

<!-- formula-not-decoded -->

The unique operator A is denoted by Df ( x ) and called the (Fr´ echet) derivative of f at x .

Definition A.6 (Second Fr´ echet derivative (Magnus &amp; Neudecker, 2019, Ch. 18)) . Let U ⊆ R m open, and consider a function f : U → R n . Suppose f is Fr´ echet differentiable at x . The second Fr´ echet derivative of f at x is the bounded bilinear map D 2 f ( x ) : R m × R m → R n defined as:

<!-- formula-not-decoded -->

Proposition A.14 (Connection to the Hessian) . If f : U → R is C 2 , then D 2 f ( x ) is symmetric (Arora et al., 2021, Thm. 5.1) and can represented by the Hessian matrix ∇ 2 f ( x ) :

<!-- formula-not-decoded -->

as noted in Magnus &amp; Neudecker 2019, Ch. 18.

Definition A.7 (Closure of a set in R p ) . Let U ⊆ R p . The closure of U , denoted U , is the smallest closed subset of R p containing U .

Definition A.8 (Euclidean balls in R p ) . Fix p ∈ N and equip R p with the Euclidean norm ∥ · ∥ 2 . For x ∈ R p and r &gt; 0 we define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In R p with the Euclidean topology one has B ( x , r ) = B ( x , r ) , i.e. the closed ball equals the topological closure of the open ball.

Definition A.9 (Second-countable subspace of R p (Munkres, 2000, §30)) . Let X ⊆ R p be equipped with the subspace topology τ X := { U ∩ X : U open in R p } . We say X is second-countable if there exists a countable family F ⊆ τ X such that every O ∈ τ X is a union of members of F . Equivalently, the countable family is a basis for τ X .

<!-- formula-not-decoded -->

Proposition A.15 (Standard facts for R p ) . Fix p ∈ N . The following hold:

1. Hausdorff (Aitken, 2020, Prop. 18) : R p with its Euclidean metric is Hausdorff.

2. Heine-Borel (Munkres, 2000, Thm. 27.3) : A subset of R p is compact iff it is closed and bounded; in particular, each closed Euclidean ball B ( x, r ) is compact.
3. Second countability (Munkres, 2000, §13 and Thm. 30.2) : R has a countable base (intervals with rational endpoints); hence R p , being a finite product of second-countable spaces, is second-countable. Moreover, subspaces of second-countable spaces are secondcountable.
4. Lindel¨ of consequence (Munkres, 2000, Thm. 30.3(a)) : Every second-countable space is Lindel¨ of; consequently, every open cover of any subspace of R p admits a countable subcover.
5. Local compactness of R p (Munkres, 2000, Thm. 29.2) : For any x ∈ R p and open neighborhood W ∋ x , there exists ε &gt; 0 with B ( x , ε ) ⊆ W , and B ( x , ε ) is compact by HeineBorel; hence R p is locally compact. Furthermore, in a Hausdorff space, local compactness is equivalent to shrinking neighborhoods with compact closures: for every neighborhood W ∋ x there exists an open V with x ∈ V ⊆ V ⊆ W and V compact.
1. f is bijective;

Definition A.10 ( C k diffeomorphism Spivak 1971, Ch. 5) . Let U, V ⊆ R p be open sets and let k ∈ N ∪ {∞} . A map f : U → V is a C k diffeomorphism if:

2. f is C k (all partial derivatives up to order k exist and are continuous);
3. the inverse map f -1 : V → U is C k .

When k = 1 we simply say diffeomorphism . Equivalently, a C k diffeomorphism is a bijective C k map whose inverse is also C k .

̸

Theorem A.2 (Inverse Function Theorem Rudin 1976, Thm. 9.24) . Let U ⊂ R p be open and f : U → R p be C 1 . Suppose a ∈ U satisfies det Df ( a ) = 0 . Then there exist open sets U 0 ⊂ U with a ∈ U 0 and V 0 ⊂ R p with f ( a ) ∈ V 0 such that

<!-- formula-not-decoded -->

is a C 1 -diffeomorphism. Moreover, the inverse f -1 : V 0 → U 0 is C 1 and

Remark 10. In Theorem A.2 we assume f : U ⊆ R p → R p , so the Jacobian Df ( a ) is a p × p (square) matrix. In this setting,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

and this is exactly the hypothesis that yields a local C 1 inverse.

Definition A.11 (Pushforward and absolute continuity (Folland, 1999, §3.2)) . Consider a Borelmeasurable map T : R p → R p and let µ be a Borel measure on R p . The pushforward measure T # µ is the Borel measure on R p defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If ν is another Borel measure on R p , we say T # µ is absolutely continuous with respect to ν , and write T # µ ≪ ν , if for every Borel set U ∈ B ( R p ) :

In particular, for Lebesgue measure Leb p , to prove T # µ ≪ Leb p for every µ ≪ Leb p , it suffices to verify that

<!-- formula-not-decoded -->

## B TRANSFORMER LANGUAGE MODEL

This appendix section gives a concise, shape-accurate specification of the decoder-only Transformer we analyze. We include it both to keep the paper self-contained and because the measure-zero arguments later hinge on architecture-dependent witnesses and exact dimension bookkeeping. We begin

with token and positional embeddings (Definition B.3), define self-attention and its causal variants (Definition B.5, Definition B.6, Definition B.7), assemble multi-head attention, layer normalization, and an MLP into a pre-LN residual block (Definition B.8, Definition B.9, Definition B.4, Definition B.11), stack L such blocks to obtain the model (Definition B.12), and conclude with the unembedding+softmax head (Definition B.10), isolating the last-token representation used in downstream proofs (Equation 29).

Definition B.1 (Token Embedding Layer) . Let V be a vocabulary, and let d ∈ N be the embedding dimension. For any input sequence s = ⟨ s 1 , . . . , s T ⟩ ∈ V ≤ K , the Token Embedding Layer is the function defined as:

<!-- formula-not-decoded -->

where E ∈ R |V|× d is a trainable embedding matrix indexed by elements of V , and E s i ∈ R d denotes the embedding vector for token s i .

This mapping is applied element-wise and is independent of the sequence length T .

Definition B.2 (Positional Embedding Layer) . Let V be a vocabulary, and let d ∈ N be the embedding dimension. For any input sequence s = ⟨ s 1 , . . . , s T ⟩ ∈ V ≤ K with T = | s | , the (learned absolute) Positional Embedding Layer is the function defined as:

<!-- formula-not-decoded -->

where P ∈ R K × d is a trainable matrix indexed by positions i ∈ [ K ] , and P i ∈ R d denotes the embedding vector for position i . This mapping depends only on positions (not on token identities) and returns the first T rows of P .

Definition B.3 (Embedding Layer) . Let V be a vocabulary, K ∈ N a context bound, and d ∈ N the embedding width. For any input sequence s = ⟨ s 1 , . . . , s T ⟩ ∈ V ≤ K with T = | s | , define the embedding layer as the sum of the token and positional embeddings:

<!-- formula-not-decoded -->

where E ∈ R |V|× d is the trainable token-embedding matrix and P ∈ R K × d is the trainable positional-embedding matrix.

Definition B.4 (Multi-Layer Perceptron) . A Multi-Layer Perceptron (MLP) with M layers is a function mlp M : R d 0 → R d M , defined recursively as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x ∈ R d 0 is the input, { W ( m ) ∈ R d m × d m -1 } M m =1 and { b ( m ) ∈ R d m } M m =1 are trainable parameters and σ is an activation function.

Definition B.5 (Self-Attention) . A Self-Attention module is a function η : R T × d in → R T × d η defined as:

,

<!-- formula-not-decoded -->

where X ∈ R T × d in is the input, Q , K , V ∈ R d in × d η are trainable parameters (query, key, and value matrices), softmax is applied row-wise, d η is the attention dimension (typically d η &lt; d in ), and T is the sequence length.

Definition B.6 (Causal Self-Attention, masked form) . Define the 'causal mask' M ∈ R T × T as:

Then, a Causal Self-Attention module is a function ˜ η : R T × d in → R T × d η , defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where X ∈ R T × d in is the input, Q , K , V ∈ R d in × d η are trainable parameters (query, key, and value matrices), softmax is applied row-wise, d η is the attention dimension (typically d η &lt; d in ), and T is the sequence length.

Definition B.7 (Causal Self-Attention, projection form) . Define the unit lower-triangular matrix L ∈ R T × T as L ij = I { j ≤ i } and consider the row normalization operation RN : D T → R T × T of Proposition A.8. Then, a Causal Self-Attention module is a function ˜ η : R T × d in → R T × d η , defined as:

<!-- formula-not-decoded -->

where X ∈ R T × d in is the input, Q , K , V ∈ R d in × d η are trainable parameters (query, key, and value matrices), RN is applied row-wise, d η is the attention dimension (typically d η &lt; d in ), and T is the sequence length.

Remark 11. Consider Z = 1 √ d η ( XQ ) ( XK ) ⊤ . Since L ii = 1 for all i ∈ [ T ] , we have that [ L ⊙ exp Z ] ii = e Z ii &gt; 0 , hence the row sum ∑ j ≤ i e Z ij ≥ e Z ii &gt; 0 and RN is well-defined.

Definition B.8 (Multi-Head Self-Attention) . A Multi-Head Self-Attention module with H heads is a function attn H : R T × d in → R T × d out , defined using the Self-Attention map from Definition B.5 or Definition B.7 with different parameter sets per head:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where { Q ( h ) , K ( h ) , V ( h ) ∈ R d in × d η } H h =1 are the head-specific parameters and W O ∈ R Hd η × d out is the output projection matrix.

Definition B.9 (Layer Normalization) . Layer Normalization is a function LN : R d → R d , defined as:

<!-- formula-not-decoded -->

where x ∈ R d is the input, µ x = 1 d ∑ d i =1 x i and σ 2 x = 1 d ∑ d i =1 ( x i -µ x ) 2 are the mean and variance of x , vectors β , γ ∈ R d are learnable parameters, and ε ∈ R + is a small constant that ensures we don't divide by zero.

Definition B.10 (Unembedding Layer) . Let V be a vocabulary and d ∈ N and U ∈ R |V|× d be a trainable projection matrix. Define the unembedding map UnEmb : R d → R |V| by

<!-- formula-not-decoded -->

Definition B.11 (Transformer Block) . A Transformer Block consists of a composition of a MultiHead Self-Attention layer with H heads (Definition B.8) and an MLP with M layers (Definition B.4), each preceded by layer normalization (Definition B.9) and wrapped with residual connections. Given an input X ∈ R T × d , the output TB( X ) ∈ R T × d is computed as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where X , H ∈ R T × d are the results of applying layer normalization row-wise to X and H , respectively, each with its own set of learnable parameters and mlp M is applied row-wise. All sub-layer parameters are dimensioned appropriately.

Definition B.12 (Transformer) . Fix L ∈ N . For each ℓ ∈ [ L ] , let TB ( ℓ ) : R T × d → R T × d denote a Transformer Block (Definition B.11) with its own parameters. Define the module

<!-- formula-not-decoded -->

Each TB ( ℓ ) maps R T × d → R T × d , so the residual additions in Definition B.11 are dimensionally valid at every depth.

Definition B.13 (Transformer Language Model) . Let V denote a finite vocabulary and K ∈ N a fixed context length. A Transformer Language Model with L layers is the composition of an embedding layer (Definition B.3), a Transformer with L blocks (Definition B.12), and an Unembedding Layer (Definition B.10).

Formally, it is a parameterized function

<!-- formula-not-decoded -->

defined as follows. Without loss of generality, consider θ = ( θ 1 ∈ R p 1 , θ 2 ∈ R p 2 , θ 3 ∈ R p 3 ) ∈ R p , which collects all the model parameters.

For an input sequence s = ⟨ s 1 , . . . , s T ⟩ with T ≤ K :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the probability of the next-token being V i is given by:

<!-- formula-not-decoded -->

Proposition B.1 (Equivalence of masked and projection causal softmax) . For any logits Z ∈ R T × T , let M and L be as in Definitions B.6-B.7. Then, row-wise,

<!-- formula-not-decoded -->

Consequently, the two definitions of the Causal Self-Attention are identical.

Proof. Fix a row i . By the mask:

<!-- formula-not-decoded -->

interpreting -∞ via a limit. On the other hand, it holds that:

<!-- formula-not-decoded -->

Therefore, L ⊙ exp Z keeps exactly the entries with j ≤ i . Then, for each row, row normalization divides the kept entries by the same positive sum ∑ k ≤ i e Z ik and leaves the others at 0 , yielding the same row as above. This holds for every row i , proving the identity.

Proposition B.2 (Embedding layer is real-analytic in the parameters) . Fix a sequence s = ⟨ s 1 , . . . , s T ⟩ ∈ V ≤ K with T = | s | . Consider the map

<!-- formula-not-decoded -->

Then this map is real-analytic on R |V|× d × R K × d (in the sense of Definition A.2).

Proof. Let S s ∈ { 0 , 1 } T ×|V| select rows { s i } T i =1 , and R T ∈ { 0 , 1 } T × K select the first T rows. Then

<!-- formula-not-decoded -->

Each map ( E , P ) ↦→ S s E and ( E , P ) ↦→ R T P is a matrix product of a constant matrix with the variable ( constant maps are real-analytic as degree0 polynomials by Proposition A.9; the product is real-analytic by Proposition A.10). Their sum is real-analytic by closure under addition (Proposition A.1). Hence ( E , P ) ↦→ Emb(s) is real-analytic.

Proposition B.3 (Joint real-analyticity of core modules and stacks) . Assume the pointwise activation σ : R → R used in the MLP is real-analytic (e.g., tanh , GELU ). Fix T ∈ [ K ] . For notational convenience define the parameter tuples

<!-- formula-not-decoded -->

Θ mlp := ( { W ( m ) , b ( m ) } M m =1 ) , Θ TB := ( Θ attn , Θ (1) LN , Θ (2) LN , Θ mlp ) , Θ Tr ,T := ( Θ (1) TB , . . . , Θ ( L ) TB ) . Then the following maps are jointly real-analytic in their inputs and parameters:

1. MLP. ( x , Θ mlp ) ↦→ mlp M ( x ) is real-analytic: each affine layer ( W , b , x ) ↦→ Wx + b is a matrix product plus addition (Proposition A.10 and Proposition A.1); the activation σ is realanalytic by assumption, and composition preserves real-analyticity (Proposition A.2). Iteration over M layers is repeated composition (Proposition A.2).
2. Layer Normalization. ( x , γ , β ) ↦→ LN( x ) = γ ⊙ x -µ x √ σ 2 x + ε + β is real-analytic: µ x and σ 2 x are (entrywise) polynomials in x (Proposition A.9); g ( x ) = σ 2 x + ε satisfies g ( x ) &gt; 0 (definition of ε &gt; 0 ), and the scalar map h ( t ) = t -1 / 2 is real-analytic on (0 , ∞ ) (classical binomial series). Thus h ◦ g is real-analytic (Proposition A.2); division by g 1 / 2 is a quotient by a nonvanishing real-analytic function (Proposition A.1); Hadamard scaling by γ and addition of β preserve realanalyticity (Proposition A.11 and Proposition A.1). Row-wise application is handled by stacking (Proposition A.12) and the vectorization equivalence (Lemma A.1).
3. Unembedding. ( h , U , γ , β ) ↦→ softmax ( U LN( h ) ) is real-analytic: LN is real-analytic by (2); multiplication by U is real-analytic (Proposition A.10); softmax is real-analytic (Proposition A.7); the overall map is a composition (Proposition A.2) and stacking across coordinates (Proposition A.12).
4. Self-Attention (vanilla or causal) and Multi-Head. Let Z = 1 √ d η ( XQ ) ( XK ) ⊤ .

(a) Vanilla SA: ( X , Q , K , V ) ↦→ softmax( Z ) XV is real-analytic by: matrix products (Proposition A.10), scaling, row-wise softmax (Proposition A.7 with stacking, Proposition A.12, and Lemma A.1), and a final matrix product.

(b) Causal SA (projection form): With L unit lower-triangular and using Definition B.7,

<!-- formula-not-decoded -->

is real-analytic: exp is real-analytic (Proposition A.5); Hadamard scaling by fixed L is realanalytic (Proposition A.11); by Remark 11, every row of L ⊙ exp( Z ) sums to a strictly positive value (the diagonal term), so the argument lies in the domain D T of Proposition A.8; hence RN is real-analytic there; the final multiplication by XV is real-analytic (Proposition A.10).

Therefore, each single attention head is real-analytic whether it is vanilla or causal (projection). For Multi-Head Self-Attention (Definition B.8), horizontal concatenation across heads is real-analytic (Proposition A.12), and the output projection by W O is a matrix product (Proposition A.10). Hence ( X , Θ attn ) ↦→ attn H ( X ) is real-analytic regardless of which attention variant each head uses.

5. Transformer Block (fixed T ). ( X , Θ TB ) ↦→ TB( X ) ∈ R T × d is real-analytic: apply LN rowwise to get X (item 2 with stacking, Proposition A.12, and Lemma A.1); apply attention (item 4) to X ; add the residual (closure under addition, Proposition A.1); apply LN row-wise to get H (item 2 with stacking and Lemma A.1); apply the row-wise MLP (item 1 with stacking, Proposition A.12); add the residual again (Proposition A.1). All intermediate matrix multiplications use Proposition A.10, and the overall structure is a composition (Proposition A.3 via Lemma A.1).
6. Transformer (fixed T ). ( X , Θ Tr ,T ) ↦→ Tr T ( X ) = TB ( L ) ◦ · · · ◦ TB (1) ( X ) is a composition of real-analytic maps from (5), hence real-analytic by Proposition A.3.

All statements extend from vector-valued to matrix-valued, row-wise applications via Proposition A.12 and Lemma A.1, and every sum/product/quotient/composition step above invokes Proposition A.1, Proposition A.10, and Proposition A.3 as indicated.

## C ALMOST SURE INJECTIVITY

This section establishes a foundational structural result: for causal Transformer Language Models with standard architectural widths and at least one attention head per block, the final hidden state at the last token is almost surely injective with respect to the input sequence, assuming the model parameters are drawn from any absolutely continuous distribution at initialization. Crucially, we show this injectivity is preserved after any finite number of gradient descent (GD) updates.

We organize the section in two parts; (i) Measure-zero collisions via real-analyticity and a witness construction and (ii) Preservation of absolute continuity under gradient descent. Each piece builds toward the main theorem, which asserts that under mild width and head assumptions, the Transformer map from input sequences to last-token representations is injective almost surely, even after multiple rounds of training. The main theorem follows.

Assumption C.1 (Minimum Embedding Dimension) . We assume the embedding dimension satisfies d ≥ 4 and d η ≥ 1 . Furthermore, we assume that each transformer block has at least one attention head. These conditions are trivially satisfied in practice: for modern large language models, embedding dimensions are typically in the hundreds or thousands, and each layer has multiple attention heads, so the assumptions impose no practical restrictions on the models under consideration.

Theorem C.1 (Finite-horizon a.s. injectivity under GD) . Fix a finite vocabulary V , a context bound K ∈ N , a time horizon T ∈ N , and consider the causal Transformer Language Model (TLM) of Definition B.13 under Assumption C.1. Let {( s t ∈ V ≤ K , p t ∈ ∆ |V|1 )} T t =1 be any sequence of samples and let { η t ∈ (0 , 1) } T t =1 be any sequence of step-sizes. Assume the parameters are randomly initialized and updated by gradient descent:

<!-- formula-not-decoded -->

where Leb p denotes Lebesgue measure on R p and L s , p : R p → R is the standard cross-entropy loss

<!-- formula-not-decoded -->

Then, with probability one over the draw of θ 0 , the last-token, last-layer representation map

<!-- formula-not-decoded -->

is injective. Equivalently,

<!-- formula-not-decoded -->

̸

where r ( · ; θ T ) denotes the last-token representation defined in Equation 29.

Proof.

Let θ 0 ∼ µ with µ ≪ Leb p . For a fixed training horizon T , define the GD update map

<!-- formula-not-decoded -->

i.e. Φ is the composition of T gradient-descent steps with step sizes { η t } T t =1 ⊂ (0 , 1) on the loss L .

- 1) Absolute continuity after T steps. By Corollary C.5.1, since µ ≪ Leb p , the pushforward law Φ # µ of θ T remains absolutely continuous:

<!-- formula-not-decoded -->

- 2) Global almost-sure distinctness. Let S := V ≤ K , which is finite. By Corollary C.2.1, under any absolutely continuous parameter law,

<!-- formula-not-decoded -->

̸

̸

Thus the map s ↦→ r (s ; θ T ) is injective almost surely, as claimed.

## C.1 ABSOLUTE CONTINUITY ENSURES ALMOST SURE INJECTIVITY

We begin by fixing two distinct sequences and asking when their last-token representations can coincide. As before, in this subsection we will consider a finite vocabulary V and a finite context window K ∈ N . Additionally, recall that for θ = ( θ 1 , θ 2 , θ 3 ) ∈ R p :

<!-- formula-not-decoded -->

and for s = t , we define the discrepancy:

<!-- formula-not-decoded -->

̸

By Proposition B.3, this map is real-analytic. To invoke the zero-set theorem, it suffices to show that h ̸≡ 0 . We construct a parameter configuration θ ⋆ such that r (s ; θ ⋆ ) = r (t ; θ ⋆ ) , treating two exhaustive cases:

̸

- Case A: If the sequences differ at their final token or in length, we isolate this distinction via selective initialization of embeddings and positional encodings.
- Case B: If they differ earlier, we construct orthogonal embeddings and exploit attention heads to differentiate the contributions to the final representation.

In both cases, we demonstrate explicit parameter settings under which the discrepancy is nonzero. This confirms h ̸≡ 0 , and the zero set { θ : r (s ; θ ) = r (t ; θ ) } has measure zero by Theorem A.1. Hence, if the parameter distribution is absolutely continuous, the probability of a collision is zero. A union bound extends this to any finite set of inputs.

Theorem C.2 (Almost-sure pairwise distinctness of last-token representations) . Let the parameter vector θ ∈ R p be drawn from any distribution absolutely continuous with respect to Lebesgue measure. Then, for any fixed s = t ,

<!-- formula-not-decoded -->

̸

Proof. Let T s = | s | and T t = | t | , and h ( θ ) := ∥ ∥ r (s ; θ ) -r (t ; θ ) ∥ ∥ 2 2 . Since h is real-analytic (Proposition B.3), it suffices to show that it is not the zero function on R p ; then h -1 ( { 0 } ) has Lebesgue measure zero by Theorem A.1, and absolute continuity transfers this to probability zero.

We construct a parameter setting θ ⋆ for which h ( θ ⋆ ) &gt; 0 , treating two exhaustive cases:

̸

Case A: T s = T t or s T s = t T t . Set all Transformer parameters to zero so that the network acts as the identity: Tr T ( X ) = X .

̸

̸

- If s T s = t T t , set E s T s = e 1 , E t T t = e 2 = e 1 , and all other rows of E to zero. Set P = 0 K × d . Then r (s ; θ ⋆ ) = e 1 , r (t ; θ ⋆ ) = e 2 , so h ( θ ⋆ ) = ∥ e 1 -e 2 ∥ 2 2 &gt; 0 .

̸

̸

- If T s = T t , set E = 0 |V|× d and P T s = e 1 , P T t = e 2 = e 1 (all others zero). Then, again, r (s ; θ ⋆ ) = e 1 , r (t ; θ ⋆ ) = e 2 , so h ( θ ⋆ ) &gt; 0 .

̸

̸

Case B: T := T s = T t and s T = t T , but s i = t i for some i ∈ [ T -1] . Let i ⋆ be the smallest such index. Note T ≥ 2 .

We construct a model with (i) all blocks after the first set to identity (zero parameters), (ii) in the first block, all heads set to zero except head 1 and the MLP is zero.

We explicitly construct embeddings and head-1 parameters ( Q , K , V ) , as well as the output projection W O , so that r (s ; θ ⋆ ) = r (t ; θ ⋆ ) .

1) Embedding Construction. Choose orthogonal vectors d

̸

⟨ e , p ⟩ = ⟨ e , q ⟩ = ⟨ p , q ⟩ = 0 , ⟨ 1 d , e ⟩ = ⟨ 1 d , p ⟩ = ⟨ 1 d , q ⟩ = 0 , ∥ e ∥ 2 = ∥ p ∥ 2 = ∥ q ∥ 2 = 1 Such vectors exist due to Assumption C.1 (requires ). Set embeddings:

<!-- formula-not-decoded -->

e , p , q ∈ R satisfying: . d ≥ 4

Thus, the input rows before LayerNorm are:

<!-- formula-not-decoded -->

- 2) LayerNorm Output. Use LayerNorm with ( γ , β ) = ( 1 , 0 ) . Since all components have zero mean, the normalization is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define:

Then:

<!-- formula-not-decoded -->

- 3) Head Parameters. Let e 1 ∈ R d η be the first standard basis vector. Set:

<!-- formula-not-decoded -->

where α, β &gt; 0 are scalars to be chosen.

Then for any j , attention vectors are:

<!-- formula-not-decoded -->

At row T , q (s) T = q (t) T = αc ep e 1 . Only the key at i ⋆ is nonzero:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Value vectors at i ⋆ differ:

<!-- formula-not-decoded -->

- 4) Attention Weights. The only nonzero score is at i ⋆ :

<!-- formula-not-decoded -->

̸

Fix δ ∈ (0 , 1 2 ) and define L := log ( 1 -δ δ ( T -1) ) . Set αβ = √ d η L/c 2 ep , so S (s) T,i ⋆ = L and S (t) T,i ⋆ &gt; L . Then:

̸

<!-- formula-not-decoded -->

- 5) Self-Attention Output.

<!-- formula-not-decoded -->

Tails are bounded by:

̸

̸

<!-- formula-not-decoded -->

̸

Since both outputs lie in span { e 1 } , we compare:

<!-- formula-not-decoded -->

Choosing δ &lt; c ep c ep +2 c e makes this strictly positive.

6) Output Projection and Propagation. Let W O be the matrix with ( W O ) 1 , 1 = 1 and all other entries zero. Then the head output is projected into coordinate 1, making the last row of the first transformer block differ between s and t in the first coordinate. Since the original rows at T were identical and the rest of the network is identity, this difference propagates to the final output, and we get r (s ; θ ⋆ ) = r (t ; θ ⋆ ) .

̸

Remark 12 (Causal Self-Attention) . The same construction works for causal self-attention. In our setup, attention at position T only needs to consider tokens at positions j ≤ T , and we only rely on attention from T to i ⋆ &lt; T . All nonzero scores occur at these allowable indices, so causal masking does not affect the computation or the argument.

Corollary C.2.1 (Almost-sure global distinctness over a finite input family) . Let S ⊆ V ≤ K be any finite collection of inputs. If θ is drawn from a law absolutely continuous w.r.t. Leb p , then

<!-- formula-not-decoded -->

̸

In particular, the last-token representations are pairwise distinct almost surely across all inputs.

̸

Proof. For each unordered pair { s , t } ⊂ S with s = t , Theorem C.2 gives Pr[ r (s ; θ ) = r (t ; θ ) ] = 0 . By the union bound over the finitely many pairs ( ( |S| 2 ) in total),

̸

<!-- formula-not-decoded -->

Hence the complement event has probability 1 .

Remark 13 (Pointwise vs. last-token injectivity) . Sutter et al. (2025) establish a related but distinct guarantee. They analyze the mapping from a prompt to the entire sequence (matrix) of hidden states, which already rules out collisions for inputs of different lengths. Their result is pointwise injectivity : if two prompts differ at position t , then the t -th hidden state (row) differs. This does not, by itself, imply injectivity of the map to the final hidden state / last-token embedding that we study, so two different prompts could still coincide at the last token-our quantity of operational interest.

## C.2 ABSOLUTE CONTINUITY OF THE PARAMETER DISTRIBUTION IS PRESERVED UNDER GD

Our goal in this subsection is to explain why absolute continuity of the parameter law at initialization survives any finite number of gradient-descent (GD) steps, thereby allowing the almost-sure injectivity argument from the previous subsection to persist throughout training. The story begins with regularity: by Proposition B.3 and Proposition A.6, the loss L s , p is real-analytic, and real-analyticity is closed under differentiation and composition. Consequently the GD map ϕ ( θ ) = θ -η ∇L s , p ( θ ) is real-analytic, its Jacobian Dϕ ( θ ) = I p -η ∇ 2 L s , p ( θ ) is real-analytic, and so is θ ↦→ det Dϕ ( θ ) (the determinant is a polynomial in the matrix entries). We then rule out the degenerate case by a witness: at θ ⋆ = 0 p , our Hessian calculation (Lemma C.4) shows det Dϕ ( θ ⋆ ) &gt; 0 , hence det Dϕ is not identically zero and its zero set C := { det Dϕ = 0 } has Lebesgue measure zero by the real-analytic zero-set theorem (Theorem A.1; summarized in Theorem C.3). On the complement R p \ C , the Inverse Function Theorem (Theorem A.2) provides, for every θ , a neighborhood on which ϕ is a C 1 diffeomorphism. Although these neighborhoods form an a priori uncountable cover, the second countability of R p (and of its subspaces) ensures a countable subcover of such charts (Proposition A.15, Lemma C.5). This countability is crucial because it lets us pass from local statements to a global measure statement via countable unions. With this cover in hand, the change-of-variables formula on each chart (Theorem C.4) implies that the image under the local inverse of any null set remains null; piecing the charts together and adding the null set C shows that preimages of Lebesgue-null sets under ϕ are null (Lemma C.6). Equivalently, ϕ pushes absolutely continuous laws to absolutely continuous laws (Theorem C.5); iterating across finitely many GD

steps preserves absolute continuity (Corollary C.5.1). Finally, combining this preservation with the almost-sure pairwise distinctness of last-token representations over any finite input family (Corollary C.2.1) yields the main consequence we need for training: the last-token representation map remains injective almost surely after any finite GD horizon.

## C.2.1 WITNESS CONSTRUCTION

LemmaC.1 (Zero-gate through scalar loss) . Let U ⊆ R m + q be open and write points as v = ( ξ , ψ ) with ξ ∈ R m and ψ ∈ R q . Let π : R m + q → R m be the projection π ( ξ , ψ ) = ξ . Consider and define f : U → R n by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let L ∈ C 2 ( R n ; R ) and set

<!-- formula-not-decoded -->

Fix v 0 = ( ξ 0 , ψ 0 ) ∈ U and assume g ( ξ 0 ) = 0 n × r . Then the Hessian of R at v 0 has block form

<!-- formula-not-decoded -->

i.e. all mixed and ψ -only second partials vanish.

Proof.

1) Introduce the bilinear multiplication map µ : R n × r × R r → R n , µ ( M , y ) = My , and the C 2 map H : U → R n × r × R r , H ( ξ , ψ ) = ( g ( ξ ) , h ( ξ , ψ )) . Then f = µ ◦ H and we write:

<!-- formula-not-decoded -->

Because µ is bilinear, Dµ ( M , y )[(∆ M , ∆ y )] = ∆ My + M ∆ y . By the chain rule:

<!-- formula-not-decoded -->

In particular, Df ( v 0 ) [ ( 0 m , · ) ] = 0 n . The second-order chain rule for Fr´ echet derivatives (e.g. Magnus &amp; Neudecker 2019, Thm. 18.4) yields:

<!-- formula-not-decoded -->

Because µ is bilinear, D 2 µ ≡ 0 and the first term is 0 . Furthermore,

<!-- formula-not-decoded -->

and it holds that:

<!-- formula-not-decoded -->

If at least one of the two directions has ξ -component zero, then D 2 g ( ξ 0 )[ h ξ , k ξ ] = 0 , so the bilinear form vanishes.

- 2) Apply the second-order chain rule to R = L◦ f at v 0 :

By ( 1 ), if at least one of the two directions is pure ψ , both terms on the right-hand side of vanish. Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Invoking Proposition A.14, this is exactly the statement that the ξψ , ψξ and ψψ Hessian blocks are 0 . The remaining block ∇ 2 ξξ R ( v 0 ) is whatever is induced by ( ⋆ ) for pairs

Lemma C.2 (Spectrum under block-diagonal extension) . Let f ∈ C 2 ( R m + q ; R ) , and fix v = ( ξ 0 , ψ 0 ) ∈ R m + q . Assume the Hessian of f at v has the block form

<!-- formula-not-decoded -->

Then the characteristic polynomial factorizes as

Consequently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i.e., the spectrum of H consists of the eigenvalues of B together with q additional zeros, and the algebraic multiplicity of the eigenvalue 0 for H equals that for B plus q .

Proof. Since H is block diagonal,

<!-- formula-not-decoded -->

The determinant of a block triangular (in particular block diagonal) matrix equals the product of the determinants of its diagonal blocks (e.g. Horn &amp; Johnson 2013, Cor. 0.8.5). Hence

<!-- formula-not-decoded -->

The zeros of χ H are the eigenvalues of H counted with algebraic multiplicity, which yields σ ( H ) = σ ( B ) ∪ { 0 } and mult H (0) = mult B (0) + q .

Remark 14. If 0 ∈ σ ( B ) , then 0 appears in σ ( H ) with multiplicity strictly larger than q ; the statement above accounts for this by adding q to the algebraic multiplicity of 0 carried over from B .

Lemma C.3 (Hessian of L w.r.t. U , β at θ ⋆ = 0 and its spectrum) . Let n := |V| and d be the embedding width. Fix (s , p ) ∈ V ≤ K × ∆ n -1 , and consider the Transformer Language Model of Definition B.13. In the unembedding layer, set the LayerNorm scale to zero, γ = 0 d . Let the parameter be ordered as

<!-- formula-not-decoded -->

Restrict attention to the ( u , β ) -coordinates and the base point

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then the Hessian of the cross-entropy loss

<!-- formula-not-decoded -->

with respect to ( u , β ) at θ ⋆ is the symmetric block matrix

<!-- formula-not-decoded -->

The spectrum of this Hessian is

<!-- formula-not-decoded -->

Proof.

- 1) Logits in vectorized form. With γ = 0 d , the LayerNorm output at the unembedding is constant: LN( h ) ≡ β (Definition B.9). Thus the logits before the final softmax are

<!-- formula-not-decoded -->

Using vec( AXb ) = ( b ⊤ ⊗ A ) vec( X ) (standard identity for vectorization, cf. Henderson &amp; Searle (1981)), with A = I n and b = β ,

<!-- formula-not-decoded -->

Therefore, near ( u , β ) = ( 0 nd , 0 d ) , the logits map is the bilinear function

<!-- formula-not-decoded -->

- 2) First and second differentials. Let ( h , η ) and ( k , ξ ) be directions in R nd × R d . Differentiating z ( u , β ) = ( β ⊤ ⊗ I n ) u gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(since both terms are multiplied by u or β ). Differentiating once more (or, equivalently, using bilinearity of z ) yields the constant symmetric bilinear form

<!-- formula-not-decoded -->

- 3) Gradient of the CE-in-softmax at the origin. Let F ( z ) := CrossEntropy(softmax( z ) , p ) . A standard computation (softmax Jacobian) gives

<!-- formula-not-decoded -->

At z = 0 n , softmax ( 0 n ) = 1 n 1 n =: b , hence

<!-- formula-not-decoded -->

- 4) Second-order chain rule for F ◦ Z at ( 0 , 0 ) . Similarly to the proof of Lemma C.1, the second differential of a composition is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used the mixed-product rule for Kronecker products and the identity

<!-- formula-not-decoded -->

- 5) Identification of the Hessian blocks. By definition of the Hessian as a bilinear form,

<!-- formula-not-decoded -->

Comparing with the expression obtained in Step 4 for arbitrary ( h , η ) and ( k , ξ ) forces

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and, because Dz ( v ) = 0 n × ( nd + d ) (so no quadratic term survives in either u or β alone),

<!-- formula-not-decoded -->

This gives exactly the claimed block matrix.

## 6) Spectrum. Let

Then

<!-- formula-not-decoded -->

The eigenvalues of ww ⊤ are ∥ w ∥ 2 2 (multiplicity 1 ) and 0 (multiplicity n -1 ); the eigenvalues of w ⊤ w equal ∥ w ∥ 2 2 (scalar). Therefore the eigenvalues of H 2 are

<!-- formula-not-decoded -->

Because H is symmetric, its eigenvalues are the real square-roots of those of H 2 , namely ±∥ w ∥ 2 (each with multiplicity d ) and 0 (with multiplicity d ( n -1) ). This is exactly the set stated in the lemma.

Lemma C.4 (Full Hessian at the witness: block form and spectrum) . Let n := |V| and d be the embedding width. Write the parameter as

<!-- formula-not-decoded -->

so p = nd +2 d + p ′ . Consider the witness point

<!-- formula-not-decoded -->

Let b := 1 n 1 n and w := b -p ∈ R n . Then the Hessian of the cross-entropy loss L ( θ ) at θ ⋆ admits the block-diagonal decomposition

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

Proof. Set γ = 0 d . Then the unembedding LayerNorm output is constant, LN( h ) ≡ β , so the logits equal z = U β . Hence, in a neighborhood of θ ⋆ , the loss depends only on ( u , β ) and is independent of ( γ , θ ′ ) .

We will apply Lemma C.1 with the open set U = R nd +2 d + p ′ , coordinates ξ = ( u , β ) and ψ = ( γ , θ ′ ) and with n = |V| , r = d . Define

<!-- formula-not-decoded -->

so that and, with L ( z ) := CrossEntropy ( softmax( z ) , p ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At the witness v 0 = ( ξ 0 , ψ 0 ) we have g ( ξ 0 ) = 0 n × d , so by Lemma C.1 all mixed and ψ -only second partials of R vanish at v 0 , i.e.

<!-- formula-not-decoded -->

Identifying R ( ξ , ψ ) ≡ L ( θ ) under the correspondence above yields

<!-- formula-not-decoded -->

Combining, Lemma C.2 and Lemma C.3, we get that

<!-- formula-not-decoded -->

Since p = nd +2 d + p ′ , the multiplicity of 0 equals p -2 d , which yields the claimed spectrum.

Theorem C.3 (GD Jacobian is nondegenerate a.e.) . Consider the setup of Theorem C.5. In particular, let ϕ : R p → R p be the one-step GD map from that theorem:

with stepsize η ∈ (0 , 1) . Then the critical set has Lebesgue measure zero in R p .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By Proposition B.3, Proposition A.6 and the closure properties of real analyticity, L s , p is real-analytic; hence so are its gradient and Hessian. Therefore ϕ is real-analytic (Lewis, 2014, Thm. 1.1.15) and

<!-- formula-not-decoded -->

It is not identically zero: at the witness θ ⋆ = 0 p , Lemma C.4 gives

Since the determinant is a polynomial in the entries, θ ↦→ det Dϕ ( θ ) is real-analytic.

<!-- formula-not-decoded -->

Hence the eigenvalues of Dϕ ( θ ⋆ ) = I p -η ∇ 2 L ( θ ⋆ ) are so

Thus det Dϕ is a nontrivial real-analytic function. By Theorem A.1, its zero set has Lebesgue measure 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.2.2 GRADIENT DESCENT PRESERVES ABSOLUTE CONTINUITY

Lemma C.5 (Countable chart cover of R p \ C ) . Consider the setup of Theorem C.5. In particular, let ϕ : R p → R p be the one-step GD map from that theorem:

with stepsize η ∈ (0 , 1) , and the measure-zero critical-set (Theorem C.3):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then there exist open sets ( U k ) k ≥ 1 covering X := R p \ C such that, for each k , the restriction ϕ k := ϕ | U k : U k → V k := ϕ ( U k ) is a C 1 diffeomorphism with C 1 inverse ψ k := ϕ -1 k .

Proof.

- 1) X is open: By Proposition B.3, Proposition A.6 and the closure rules of real-analyticity, L s , p is C 2 , hence ϕ is C 1 . The map θ ↦→ Dϕ ( θ ) is continuous, and the determinant is a continuous polynomial in the entries, so g ( θ ) := det Dϕ ( θ ) is continuous. Therefore C = g -1 ( { 0 } ) is closed (Rudin, 1976, Thm. 4.8) and X = R p \ C is open.

̸

- 2) Local diffeomorphisms by the Inverse Function Theorem: Fix θ ∈ X . Then g ( θ ) = 0 , so by the Inverse Function Theorem (Theorem A.2) there exist open neighborhoods U θ ∋ θ and V θ ∋ ϕ ( θ ) such that

is a C 1 diffeomorphism with C 1 inverse ψ θ := ϕ -1 θ . Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular Dϕ ( x ) is invertible for all x ∈ U θ , whence U θ ⊂ X . Thus { U θ } θ ∈ X is an open cover of X by IFT charts.

- 3) Select a countable subcover: By Proposition A.15(3), R p is second-countable; subspaces of second-countable spaces are second-countable, hence X is second-countable. By Proposition A.15(4), every open cover of a second-countable space admits a countable subcover. Therefore there exist points θ 1 , θ 2 , . . . ∈ X such that X = ⋃ ∞ k =1 U θ k .

Set U k := U θ k , V k := V θ k , and ϕ k := ϕ | U k = ϕ θ k , ψ k := ψ θ k . Each ϕ k is a C 1 diffeomorphism with C 1 inverse ψ k by Step 2. This yields the desired countable chart cover of X .

Theorem C.4 (Change of Variables Folland 1999, Thm. 2.47(b)) . Let U , V ⊆ R p be open and ψ : V → U a C 1 diffeomorphism. If E ⊆ V is Lebesgue measurable, then

<!-- formula-not-decoded -->

Lemma C.6 (Pre-images of null sets are null) . Consider the setup of Theorem C.5, in particular the C 1 gradient descent map:

<!-- formula-not-decoded -->

and its critical set C := { θ ∈ R p : det Dϕ ( θ ) = 0 } . Then, for every measurable A ⊆ R p ,

<!-- formula-not-decoded -->

Proof. Let X = R p \ C and decompose the pre-image:

<!-- formula-not-decoded -->

The first set is contained in C , a measure zero set (Theorem C.3), hence has Leb p -measure 0 . By Lemma C.5, cover X by countably many charts { U k } on which ϕ k := ϕ | U k is a C 1 diffeomorphism onto V k := ϕ ( U k ) with inverse ψ k ∈ C 1 ( V k ; U k ) . Then, it holds that:

<!-- formula-not-decoded -->

Since Leb p ( A ) = 0 and both A and V k are measurable, A ∩ V k is measurable and has measure 0 . By Theorem C.4 applied to ψ k with E = A ∩ V k ,

<!-- formula-not-decoded -->

Therefore, each ϕ -1 ( A ) ∩ U k is null and because a countable union of null sets is null, it holds that:

<!-- formula-not-decoded -->

Theorem C.5 (Preservation of absolute continuity under one GD step) . Fix a finite vocabulary V , a context bound K ∈ N , and the Transformer language model f of Definition B.13. For any sample (s , p ) ∈ V ≤ K × ∆ |V|1 and any learning rate η ∈ (0 , 1) , let ϕ : R p → R p be the gradient-descent update, defined as:

<!-- formula-not-decoded -->

where L s , p : R p → R is the standard Cross Entropy loss:

<!-- formula-not-decoded -->

Then, gradient-descent preserves absolute continuity: for every absolutely continuous probability law µ on R p , its image under ϕ remains absolutely continuous:

<!-- formula-not-decoded -->

Therefore, the updated parameters θ ′ := ϕ ( θ ) are absolutely continuous.

Proof. By Proposition B.3 and closure properties, L s , p is C 2 , hence ϕ ∈ C 1 and is Borelmeasurable. From Theorem C.3 the critical set

<!-- formula-not-decoded -->

has Leb p -measure 0 . Therefore, the hypothesis of Lemma C.6 holds, and we have the property:

<!-- formula-not-decoded -->

Let A be any Borel set with Leb p ( A ) = 0 . Then

<!-- formula-not-decoded -->

because µ ≪ Leb p and Leb p ( ϕ -1 ( A ) ) = 0 by ( † ) . Since this holds for every Leb p -null set A , we conclude ϕ # µ ≪ Leb p .

Corollary C.5.1 (Preservation of absolute continuity under finitely many GD steps) . Fix a finite vocabulary V , a context bound K ∈ N , and the Transformer language model f of Definition B.13. For t = 1 , . . . , T , let (s t , p t ) ∈ V ≤ K × ∆ |V|1 and η t ∈ (0 , 1) , and define the t -th GD update

<!-- formula-not-decoded -->

Let the T -step update map be the composition

<!-- formula-not-decoded -->

Then, for every absolutely continuous probability law µ on R p , its image under Φ remains absolutely continuous:

<!-- formula-not-decoded -->

Equivalently, if θ (0) ∼ µ with µ ≪ Leb p and

<!-- formula-not-decoded -->

then the T -step parameters θ ( T ) = Φ ( θ (0) ) are absolutely continuous.

Proof. Since the result of Lemma C.6 holds for each ϕ t , for any null set A , repeated preimages remain null:

<!-- formula-not-decoded -->

The same argument as in the proof of Theorem C.5 then yields the claim.

## D LEFT-INVERTIBILITY VIA SIP-IT

Goal. We study when and how the hidden states of a causal decoder-only Transformer admit a left inverse : given the layerℓ representation at position t and the true prefix π = s 1: t -1 , can we recover the next token s t ?

Main idea. Under mild randomness in the parameters and causal masking, the one-step last-token map that sends a candidate token v to the layerℓ representation at position t (conditioning on π ) is almost-surely injective, and in fact has a positive separation margin. This yields a simple verifier: declare v correct iff the observed hidden state lies in a small ball around F ( v ; π, t ) .

Algorithmic consequence. Because causality localizes the dependence to ( π, s t ) , we can invert an entire sequence sequentially with a single pass over the vocabulary per position. We call this procedure SIP-IT (Sequential Inversion via Prefixwise Injective Tests), and we show exact (and robust) recovery holds almost surely, with worst-case time Θ( T |V| ) .

Standing conventions for this section. Fix a layer index ℓ ∈ [ L ] . For any input sequence s = ⟨ s 1 , . . . , s T ⟩ , define the layer outputs row-wise by

<!-- formula-not-decoded -->

and write h t (s) to denote the row of H ( ℓ ) (s) at position t . Furthermore, we use ⊕ for sequence concatenation: if s = ⟨ s 1 , . . . , s t -1 ⟩ and v ∈ V , then s ⊕ v = ⟨ s 1 , . . . , s t -1 , v ⟩ .

The parameters θ and target layer ℓ are considered fixed and omitted for simplicity.

Assumption D.1 (Causal self-attention throughout) . Every attention layer in every block is causal in the sense of Definitions B.6/B.7. Consequently, for any s and any t ∈ [ T ] ,

<!-- formula-not-decoded -->

Assumption D.2 (Injectivity Assumption) . SIP-IT is applied to models initialized with parameters drawn from an absolutely continuous distribution and trained via (mini-batch) gradient descent with step sizes in (0 , 1) , as described in Appendix C. Under these conditions, any network considered in the sequel is almost-surely injective (Theorem C.1).

## D.1 ONE-STEP LAST-TOKEN MAPS

Wefirst isolate the positionwise map that drives inversion. Fix a position t and prefix π ∈ V t -1 . The one-step map F ( · ; π, t ) sends a candidate token v to the layerℓ hidden state at position t obtained when the prefix is π and the token at t is v . Causality implies that h t depends only on ( π, v ) (not on any future tokens), and we show that, for almost all parameter settings, F is injective with a strictly positive pairwise margin over V .

Definition D.1 (One-step map at time t under prefix π ) . Let π ∈ V t -1 be a fixed prefix (possibly t = 1 , when π is empty). Define

<!-- formula-not-decoded -->

Remark 15. F is simply a function that returns the hidden output of token v at the ℓ transformer block given that π is used a fixed prefix. This map allows us to have a convenient notation for introducing results about inversion. Furthermore, since F is built using ℓ transformer blocks, it is parameterized by θ . Nevertheless, for the sake of simplicity, we will refer to F ℓ, θ simply as F .

Once the One-step map (Definition D.1) is introduced, one can present its a.s. injectivity through an application of the previously obtained result (Theorem C.1). Furthermore, one can deploy the common prefix to introduce a stronger notion of injectivity: margin separation (Lemma D.1).

Theorem D.1 (A.s. one-step injectivity) . Fix t and the prefix π ∈ V t -1 . Under Assumptions D.1 and D.2, it holds that:

<!-- formula-not-decoded -->

̸

Equivalently, F is injective almost-surely.

Then,

Proof. Set the finite family S t,π := { π ⊕ v : v ∈ V} ⊆ V t and view h t (s) as the last-token representation of the truncated Transformer consisting of the first ℓ blocks. All assumptions used in Corollary C.2.1 remain valid for this truncated model. Applying the corollary with S = S t,π yields, almost-surely, h t ( π ⊕ v ) = h t ( π ⊕ v ′ ) whenever v = v ′ . This is exactly the injectivity of F .

̸

̸

Lemma D.1 (Strict separation margin a.s.) . Under the conditions of Theorem D.1, define the (datadependent) margin

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By Theorem D.1, with probability 1 the set

<!-- formula-not-decoded -->

consists of |V| distinct points in R d . On this event of full probability, every pairwise distance among these finitely many points is strictly positive, so their minimum is strictly positive as well.

Thus, the event { ∆ π,t &gt; 0 } coincides with the event that F is injective on V . Since injectivity holds almost-surely by assumption, we conclude that Pr[∆ π,t &gt; 0] = 1 .

## D.2 THE CORE ROUTINES: LOCAL VERIFIERS, ACCEPTANCE REGIONS, AND POLICIES

Given F ( · ; π, t ) , inversion reduces to a local hypothesis test: for an observed ̂ h t , which token's predicted representation is closest? We formalize this with acceptance regions -closed balls around F ( v ; π, t ) -and a verifier that accepts v iff ̂ h t lies in its ball. Almost-sure injectivity yields uniqueness at radius 0 , and a positive margin yields uniqueness for any ε &lt; ∆ π,t / 2 . To explore candidates efficiently, we couple the verifier with any policy that enumerates untried tokens (e.g., uniform without replacement or a gradient-guided ranking).

Definition D.2 (Local verifier and acceptance tolerance) . Given a tolerance ε ≥ 0 , define the acceptance region for symbol v as the closed ball (Definition A.8):

<!-- formula-not-decoded -->

A candidate token v ∈ V is verified for observation ̂ h t if and only if ̂ h t ∈ A π,t ( v ; ε ) .

Remark 16 (Decoding via acceptance regions) . Given a prefix π ∈ V t -1 and the observation ̂ h t at position t , we identify the next token by checking in which acceptance region ̂ h t lies: declare v verified iff ̂ h t ∈ A π,t ( v ; ε ) . By Lemma D.1, for any ε &lt; ∆ π,t / 2 the regions {A π,t ( v ; ε ) } v ∈V are pairwise disjoint; hence there is at most one verified token (and in the noiseless case ε = 0 , exactly one).

Building on the intuition in Remark 16, we introduce two radii to define acceptance regions that avoid collisions:

Proposition D.1 (Probabilistic soundness and uniqueness of the local verifier) . Fix position t and prefix π ∈ V t -1 . Under Assumptions D.1 and D.2, for all v ⋆ ∈ V , the following hold with probability one:

1. Noiseless soundness. If ε = 0 and ̂ h t = F ( v ⋆ ; π, t ) , then v ⋆ is the unique verified symbol.
2. Robust uniqueness. If ε &lt; ∆ π,t / 2 and ̂ h t ∈ A π,t ( v ∗ ; ε ) , then v ⋆ is the unique verified symbol.

Proof. Recall that under Assumptions D.1 and D.2, F is injective and ∆ π,t &gt; 0 almost-surely.

̸

(1) Noiseless soundness. For any v ∈ V , A π,t ( v ; 0) = { F ( v ; π, t ) } . If ̂ h t = F ( v ⋆ ; π, t ) and some v = v ⋆ were also verified at ε = 0 , we would have F ( v ; π, t ) = F ( v ⋆ ; π, t ) , which is a probability zero event under the assumptions made. Hence v ⋆ is uniquely verified almost-surely.

̸

<!-- formula-not-decoded -->

- (2) Robust uniqueness. Assume ε &lt; ∆ π,t / 2 and ∥ ̂ h t -F ( v ⋆ ; π, t ) ∥ 2 &lt; ε . If some v = v ⋆ were also verified, then ∥ ̂ h t -F ( v ; π, t ) ∥ 2 ≤ ε . By the triangle inequality,

contradicting the definition of ∆ π,t (again, valid under the assumptions made). Thus v ⋆ is uniquely verified almost-surely.

Finally, we introduce the last conceptual block required to build the inversion algorithm:

Definition D.3 (Policy algorithm) . Let V be a finite vocabulary. A policy algorithm is a (possibly randomized) map

<!-- formula-not-decoded -->

Remark 17 (Enumeration property) . Intuitively, a policy chooses any token not tried yet. Starting from C 0 = ∅ and iterating

(When C = V the map is undefined.)

<!-- formula-not-decoded -->

produces a sequence ( v 1 , . . . , v |V| ) that is a (possibly random) permutation of V . Thus, in exactly |V| steps, every token is output once with no repetitions.

Two examples of policy algorithms. We give (i) a uniform-random without replacement policy and (ii) a gradient-guided policy.

## Algorithm 2 Policy (Random)

Require: Vocabulary V ; visited set C ; embedding matrix E ∈ R |V|× d

- Ensure: Next token ID and embedding 1: Sample a permutation L = ( v 1 , . . . , v |V| ) uniformly from V 2: Define ρ ( v ; π ) as the rank of v in L 3: v ⋆ = arg min v ∈V\ C ρ ( v ; π ) 4: return v ⋆ , E v ⋆

## Algorithm 3 Policy (Gradient-based)

Require: Vocabulary V ; visited set C ; embedding matrix E ∈ R |V|× d ; prefix π ∈ V t -1 ; layer ℓ ; previous continuous embedding e ( j -1) ; step size γ &gt; 0 ; gradient-based update rule G Ensure: Next token ID and embedding

<!-- formula-not-decoded -->

- 3: Get L = ( v 1 , . . . , v |V| ) by ordering v i based on ℓ 2 ( E v i , e ( j ) )
- 2: e ( j ) ←G ( e ( j -1) , g , γ )
- 4: Define ρ ( v ; π ) as the rank of v in L
- 5: v ⋆ = arg min v ∈V\ C ρ ( v ; π )
- 6: return v ⋆ , e ( j )

Remark 18 (Bypassing the embedding layer) . We slightly overload notation and write F ( e ; π, t ) . Here we bypass the token embedding lookup and inject a continuous vector at the current position: the first t -1 rows of H (0) are set to Emb( π ) and the t -th row is set to e . This extension is used only to guide the search (e.g., in Policy-Gradient ). All theoretical guarantees are stated for F ( v ; π, t ) with v ∈ V and are unaffected by allowing F to accept a continuous proxy during candidate scoring. Any extra inputs/side outputs used by a policy (such as the updated proxy) are orthogonal to the correctness statements.

Remark 19 (Practical choice of policy) . Both Alg. 2 and Alg. 3 satisfy Definition D.3. In practice we use the gradient-guided policy with standard gradient descent updates, as it tends to find the verified token with far fewer proposals: the next token is chosen by ranking V by the distance ∥ E v -e ( j ) ∥ 2 to the updated proxy e ( j ) . This preserves the same worst-case guarantees (single pass over V ) while improving empirical efficiency.

## D.3 GLOBAL INVERSION VIA SIP-IT

We now compose the local verifier into a sequential decoder. At step t , causality ensures h t (s) = F (s t ; π, t ) for the true prefix π = s 1: t -1 . Since the verifier uniquely accepts s t (noiselessly, and robustly under perturbations below half the margin), any covering policy must encounter and accept the true token within a single pass over V . Iterating from t = 1 to T yields exact recovery almost surely; we also quantify robustness and the worst-case runtime.

We are now ready to introduce our inversion algorithm: SIP-IT (Alg. 1). The algorithms applies to decoder-only transformers with causal self-attention (Assumption D.1), and assumes injectivity, which occurs with almost-surely (Assumption D.2). We assume access to the layerℓ hidden states per position { ̂ h t } T t =1 and to the parameters needed to evaluate the local verifier from Definition D.2 for arbitrary ( t, π, j ) , as well as the gradient (when needed), namely to the model up to layer ℓ . A policy algorithm is fixed (e.g., Alg. 3).

We begin by recording the following standard lemma and omitting the proof, as it is immediate from causal masking: under causal self-attention, the representation at position t is independent of future tokens.

Lemma D.2 (Causal factorization and prefixwise identifiability) . Under Assumptions D.1 and D.2, fix position t ∈ [ T ] . For any s = ⟨ s 1 , . . . , s T ⟩ with π = ⟨ s 1 , . . . , s t -1 ⟩ ,

<!-- formula-not-decoded -->

where F is the one-step map from Definition D.1.

Proof. With causal masking, position t attends only to positions ≤ t . Evaluating the network up to layer ℓ therefore yields a representation at t that is a function of the prefix π and the current token s t only, i.e. F (s t ; π, t ) , as claimed.

Proposition D.2 (The verifier is the right primitive) . Fix t and a true prefix π = ⟨ s 1 , . . . , s t -1 ⟩ . Under Assumption D.1, the observed hidden state at step t satisfies h t (s) = F (s t ; π, t ) (Lemma D.2). In addition, under Assumption D.2, F is injective and has positive margin ∆ π,t &gt; 0 almost-surely (Theorem D.1 and Lemma D.1). Consequently, for the local verifier of Definition D.2, the following hold with probability one:

1. ( Noiseless ) With ε = 0 and observation ̂ h t = h t (s) , the unique verified token is s t .
2. ( Robust ) If ̂ h t = h t (s) + e t with ∥ e t ∥ 2 &lt; ε &lt; ∆ π,t / 2 , then s t is the unique verified token.

Proof. Immediate from Lemma D.2 and Proposition D.1 applied with v ⋆ = s t , which holds almostsurely by Theorem D.1 and Lemma D.1.

Proposition D.3 (Eventual acceptance under increasing enumeration) . Fix a position t and the true prefix π = ⟨ s 1 , . . . , s t -1 ⟩ . Under Assumption D.1 and Assumption D.2, let ε ≥ 0 and work on the probability-one event where the local verifier uniquely accepts the true token s t (e.g., ε = 0 or ε &lt; ∆ π,t / 2 ; see Proposition D.2).

Let Π be any policy algorithm (Definition D.3). Define the increasing visited sets by C 0 = ∅ , v i := Π( C i -1 ) , and C i := C i -1 ∪ { v i } for i ≥ 1 , and stop at the first index

<!-- formula-not-decoded -->

Then ( v i ) i ≥ 1 enumerates V without replacement and τ ≤ |V| almost surely. In particular, for the fixed prefix π , the policy's increasingly expanding search over V eventually proposes the unique verified token s t and accepts it with probability 1 .

Proof. Work on the probability-one event of Proposition D.2 (under Assumption D.1 and Assumption D.2 with the stated ε ), on which the local verifier at step t uniquely accepts the true token s t . Equivalently,

<!-- formula-not-decoded -->

Enumeration without replacement. By the definition of a policy algorithm (Definition D.3), v i = Π( C i -1 ) ∈ V \ C i -1 and C i = C i -1 ∪{ v i } . Hence v i / ∈ C i -1 and |C i | = |C i -1 | +1 . Inducting on i yields that ( v i ) i ≥ 1 has no repetitions and C i contains exactly i distinct tokens. Since V is finite, after |V| steps we have C |V| = V , i.e., ( v i ) |V| i =1 is a permutation of V (this holds pathwise, for any realization of the policy's internal randomness).

Eventual acceptance. Because ( v i ) is a permutation of V , there exists a unique index j ∈ { 1 , . . . , |V|} with v j = s t . By equation 35,

<!-- formula-not-decoded -->

so τ ≤ |V| and the process accepts s t .

Since the event on which equation 35 holds has probability 1 , the conclusion (eventual acceptance at finite τ ) holds almost surely.

Theorem D.2 (Correctness of SIP-IT (noiseless &amp; robust)) . For each t ∈ { 1 , . . . , T } let π t = ⟨ s 1 , . . . , s t -1 ⟩ and let ∆ π t ,t &gt; 0 be the margin of the one-step map F ( · ; π t , t ) from Lemma D.1. Under Assumptions D.1 and D.2, run SIP-IT (Alg. 1) with a tolerance ε ≥ 0 and observations

<!-- formula-not-decoded -->

where the perturbations satisfy ∥ e t ∥ 2 ≤ ε for all t and

<!-- formula-not-decoded -->

Then, with probability 1 over the model parameters: (i) for every t , the inner for-loop over j (the loop over vocabulary candidates) terminates within |V| iterations by accepting the true token s t ; and (ii) after the outer for-loop over t (the loop over positions) finishes, the algorithm outputs the exact sequence ̂ s = s .

In particular, this covers the noiseless case by taking ε = 0 and ̂ h t = h t (s) , and the robust case with any uniform ε such that max t ∥ e t ∥ 2 ≤ ε &lt; 1 2 min t ∆ π t ,t .

Proof. By Assumption D.2, Theorem D.1, and Lemma D.1, there is a probability-one event on which, for all t , F ( · ; π t , t ) is injective with strictly positive margin ∆ π t ,t . Intersecting across finitely many t preserves probability 1. Work on this event.

By Assumption D.1 and Lemma D.2, h t (s) = F (s t ; π t , t ) . Since ∥ e t ∥ 2 ≤ ε ,

<!-- formula-not-decoded -->

so the local verifier accepts s t . Moreover, because ε &lt; 1 2 ∆ π t ,t , Proposition D.1(2) implies robust uniqueness :

<!-- formula-not-decoded -->

When ε = 0 , equation 36 also holds by Proposition D.1(1). We now analyze SIP-IT and proceed by induction on t .

Base case ( t = 1 ). The outer for-loop over t begins with ̂ s = ⟨ ⟩ = π 1 . Inside the inner forloop over j (the loop over vocabulary candidates), the policy (Definition D.3) enumerates V without replacement. By Proposition D.3, there exists j ⋆ ≤ |V| such that v j ⋆ = s 1 , which is accepted and triggers the break ; the algorithm appends s 1 .

Inductive step. Suppose after completing the inner loop at step t -1 the algorithm has appended s t -1 , so the prefix entering step t is ̂ s = π t . By equation 36, within the inner loop the verifier accepts exactly when v j = s t . Because the policy enumerates V without replacement, some j ≤ |V| satisfies v j = s t , which is accepted, appended, and the inner loop break s.

Thus for every t , the inner loop terminates by accepting s t within |V| iterations, and after the outer loop finishes we have appended (s 1 , . . . , s T ) , i.e., ̂ s = s . Since the reasoning holds on a probabilityone event (independent of the policy's internal randomness), the conclusion is almost sure.

gpt2

Figure 7: Seeking collisions in a large-scale prompt set (§4.1). For each layer, boxplots show the distribution (log scale) of the minimum pairwise ℓ 2 distances between last-token states across prompts for the GPT-2 model family ( Small , Medium , and Large ); red bars mark medians and the dashed line indicates the collision threshold 10 -6 .

<!-- image -->

Proposition D.4 (Termination and linear step bound) . Run SIP-IT (Alg. 1) on a lengthT sequence with any policy that enumerates V without replacement. Then the algorithm halts after a finite number of iterations. Moreover, in the worst case the inner for-loop over j executes at most |V| iterations at each position t , so the total number of verifier tests across the entire run is at most T |V| . In particular, the number of loop iterations grows linearly with T · |V| .

Proof. Fix a position t . The inner for-loop over j proposes unvisited tokens and stops when a candidate verifies, or after exhausting V . Because the policy enumerates without replacement, the loop can execute at most |V| iterations at step t . The outer for-loop over t runs for exactly T positions, hence the total number of inner-loop iterations (i.e., verifier tests) is at most ∑ T t =1 |V| = T |V| &lt; ∞ . Therefore the algorithm halts and the total number of tests is linear in T · |V| .

Remark 20 (Iterations vs. wall-clock time) . Proposition D.4 bounds the number of iterations/tests : the inner loop performs at most |V| verifier tests per position, so the total is Θ( T |V| ) . This is an iteration complexity statement that holds for any policy satisfying the 'enumerate V without replacement' property. Actual wall-clock time also depends on the per-test cost (one call to F ( v ; π, t ) plus a distance) and on any policy overhead (e.g., forward/backward proxy updates, scoring, sorting). A generic decomposition is

<!-- formula-not-decoded -->

where C test is the cost of one membership test and C policy ( t ) captures policy-specific work at step t . Thus, if |V| is treated as fixed and C test , C policy ( t ) are bounded (e.g., a constant number of proxy updates and at most one ranking per update), wall-clock time is O ( T ) . If |V| grows or the policy sorts per update, additional factors like |V| or log |V| may appear in the time, but the termination and the Θ( T |V| ) iteration bound remain unchanged.

Remark 21 (Choosing the tolerance ε ) . Theory guarantees uniqueness whenever ε &lt; 1 2 ∆ π,t (Proposition D.1). Since ∆ π,t is unknown, two practical choices work well: (i) backoff : start with

gemma-3-1b-pt

Figure 8: Seeking collisions in a large-scale prompt set (§4.1). For each layer, boxplots (log scale) show the distribution of minimum pairwise ℓ 2 distances between last-token states across prompts for the Gemma-3 model family (1B, 4B, 12B); red bars denote medians and the dashed line marks the collision threshold 10 -6 .

<!-- image -->

a small ε and increase only if no token verifies; (ii) calibration : set ε from held-out hidden states at layer ℓ . In all cases the decision rule remains a simple yes/no membership test.

Remark 22 (Why SIP-IT is sequential) . The algorithm never solves a global assignment. At position t it conditions on the current prefix π and queries the local verifier for a single token. Causality (Assumption D.1) ensures h t depends only on ( π, s t ) , so these local, prefixwise decisions compose to recover the full sequence.

## E ADDITIONAL EXPERIMENTS AND IMPLEMENTATION DETAILS

## E.1 IMPLEMENTATION DETAILS

SIP-IT implementation. We implement SIP-IT exactly as in Alg. 1 with the gradient-guided policy. To stabilize the continuous proxy used for ranking, we periodically project it back to the nearest token embedding every K =50 candidate proposals:

<!-- formula-not-decoded -->

without taking gradients through this projection. This heuristic affects efficiency only; the verifier and all correctness guarantees remain unchanged.

HARDPROMPTS implementation. The original HARDPROMPTS method Wen et al. (2023) targets multimodal vision-language models and optimizes prompts via a CLIP-based similarity objective. In our text-only setting we lack the vision branch and CLIP loss, so we adapt Algorithm 1 of Wen et al. (2023) to language models by replacing the objective with the same ℓ 2 loss used in SIP-IT's gradient calculation, and setting the optimization steps T = 1 4 # tokens · |V| . All other details (step sizes, stopping rules) mirror our SIP-IT setup to ensure a fair comparison.

## E.2 ADDITIONAL ABLATIONS

We report three complementary ablations that probe how separation behaves across depth, length, and model family.

GPT-2 family across depth. For GPT-2 Small , GPT-2 Medium , and GPT-2 Large , the per-layer boxplots (log scale) of the minimum pairwise ℓ 2 distances between last-token states in Figure 7 show that all minima sit orders of magnitude above the collision threshold 10 -6 at every depth, and the typical separation increases with depth (median red bars drift upward). This rules out collisions in practice and indicates that deeper blocks monotonically sharpen last-token distinctions in these models.

Figure 9: Sequence length versus distance over all pairs of distinct prompts for Gemma-1B .

<!-- image -->

Gemma-3 family across depth and scale. Across Gemma3-1B , Gemma3-4B , and Gemma3-12B , the layerwise boxplots (log scale) in Figure 8 again show minima far above 10 -6 at all depths. Both depth and model size trend positively with separation: medians and lower whiskers move upward in deeper layers and larger models, indicating progressively stronger margins and no observed collisions.

Effect of sequence length (Gemma-1B). Varying the prompt length reveals that min/mean/max pairwise distances rise quickly for short sequences and then plateau, with the minimum never approaching zero (see Fig- ure 9). This suggests that beyond a modest context size, additional tokens do not erode separability; margins stabilize rather than collapse, making collisions unlikely for any prompt length explored.

Overall, these ablations corroborate the main text: last-token states remain well-separated across architectures and depths, separation typically grows with depth (and scale for Gemma), and margins stabilize with sequence length, aligning with our almost-sure injectivity guarantees and with SIPIT's exact recovery behavior.