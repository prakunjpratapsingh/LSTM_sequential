LSTM Many to One

Architecture (Many-to-One LSTM — MNIST)

Treated images as sequences of patches (patch size 4×4, seq len 64) andreshape to a token sequence (B×L×16).


Built an input projection by applying  Linear layer + ELU to each token (Linear(16→32)), yielding embedded tokens (B×L×32).


Implemented a stacked LSTM (3 layers, hidden=32) with batch_first processing for stable sequence learning over the token sequence, per-step outputs (B×L×32) and final states (h_T, c_T) ∈ (3,B,32).


Added residual MLP blocks (LayerNorm → Linear → ELU → Linear + skip) to refine timestep features before classification.


Used a many-to-one head: select last LSTM output → Linear(32→10) for digit logits.


Achieved 97.01% test accuracy.
