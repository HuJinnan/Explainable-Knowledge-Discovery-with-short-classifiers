# Formal Pattern-Structure Definition

## 1. Representation of one patient

Each patient is represented by:

- a **static context**
- and a **temporal sequence of states**

So one patient is:

\[
\langle S_1, S_2, \dots, S_T \rangle
\]

where each \(S_t\) corresponds to one time window.

The static context contains:

- age bin
- gender
- cancer type
- cancer subtype code when relevant
- normalized stage
- metastasis flag

The dynamic part is the sequence of states.

## 2. Definition of one state

Each state is a tuple of feature descriptions:

\[
S = (x_{HB}, x_{WBC}, x_{PLT}, x_{NEUT}, x_{CREA}, x_{UREA}, x_{ALT}, x_{AST}, x_{BILI}, x_{ALB}, x_{LDH}, x_{ESR}, x_{ECOG}, x_{ADM}, x_{CHEMO}, x_{RADIO})
\]

where:

- laboratory features are interval-valued
- ECOG is interval-valued
- admissions is a non-negative integer
- chemotherapy and radiotherapy are binary

## 3. Feature-level pattern structures

### 3.1 Numerical interval features

This includes:

- all selected laboratory tests
- ECOG

Each observed numeric value \(v\) is represented as the point interval \([v, v]\).

For two intervals \([a_1, b_1]\) and \([a_2, b_2]\), define similarity by:

\[
[a_1, b_1] \sqcap [a_2, b_2] = [\min(a_1, a_2), \max(b_1, b_2)]
\]

Subsumption is:

\[
[a_1, b_1] \sqsubseteq [a_2, b_2]
\iff
[a_1, b_1] \sqcap [a_2, b_2] = [a_1, b_1]
\]

Thus \([a_1,b_1]\) is more general than \([a_2,b_2]\) iff it contains it.

### 3.2 Missing numerical values

If a numerical feature is missing in a state, it is encoded by the **top interval** of that feature, i.e. the largest admissible interval for that feature.

This is done for mathematical correctness of the pattern structure.

Additionally, a provenance flag is stored to indicate that this top interval comes from missingness rather than from observation.

### 3.3 Admissions

Admissions is modeled with a Min Pattern Structure.

For two values \(a,b \in \mathbb{N}_0\):

\[
a \sqcap b = \min(a,b)
\]

and

\[
a \sqsubseteq b \iff a \sqcap b = a
\]

Hence:

\[
a \sqsubseteq b \iff a \leq b
\]

So smaller values are more general, and a pattern value \(k\) means:

- at least \(k\) admissions

In particular:

- \(0\) is the top element

### 3.4 Binary features

For binary features such as:

- `chemo_active`
- `radiotherapy_active`

we use presence-style semantics:

- \(1\) means present
- \(0\) is the more general value

Similarity is:

\[
1 \sqcap 1 = 1
\]

and in all other cases:

\[
x \sqcap y = 0
\]

Subsumption is:

\[
0 \sqsubseteq 0,\quad 0 \sqsubseteq 1,\quad 1 \sqsubseteq 1
\]

but:

\[
1 \not\sqsubseteq 0
\]

So if a binary feature is present in a pattern, it must also be present in the matched object.

## 4. State pattern structure

The state pattern structure is the direct product of the feature-level pattern structures.

For two states:

\[
S^{(1)} = (x^{(1)}_1, \dots, x^{(1)}_p), \qquad
S^{(2)} = (x^{(2)}_1, \dots, x^{(2)}_p)
\]

their similarity is computed componentwise:

\[
S^{(1)} \sqcap S^{(2)} =
(x^{(1)}_1 \sqcap x^{(2)}_1,\dots,x^{(1)}_p \sqcap x^{(2)}_p)
\]

Subsumption is also componentwise:

\[
S^{(1)} \sqsubseteq S^{(2)}
\iff
\forall j,\; x^{(1)}_j \sqsubseteq x^{(2)}_j
\]

So a state is more general than another state if every one of its feature descriptions is more general in the same direction.

## 5. Sequence representation

A patient trajectory is a finite sequence of states:

\[
T = \langle S_1, S_2, \dots, S_n \rangle
\]

In this work we restrict attention to **contiguous subsequences**.

## 6. Contiguous subsequence order

Let:

\[
T_1 = \langle A_1, \dots, A_k \rangle,\qquad
T_2 = \langle B_1, \dots, B_m \rangle
\]

with \(k \le m\).

We say that \(T_1\) is a contiguous subsequence of \(T_2\), written:

\[
T_1 \sqsubseteq T_2
\]

iff there exists an index \(j\) such that:

\[
\langle B_j, B_{j+1}, \dots, B_{j+k-1} \rangle
\]

is a contiguous block of \(T_2\), and for every position \(i = 1,\dots,k\):

\[
A_i \sqsubseteq B_{j+i-1}
\]

Thus the pattern sequence is more general than a matched contiguous block in the patient trajectory.

## 7. Sequence similarity

Similarity between two sequences is defined in the same spirit as the sequence pattern-structure paper.

For all possible contiguous alignments between two sequences:

1. match aligned contiguous blocks
2. compute the statewise meet position by position
3. collect all resulting common subsequences
4. keep only the **maximal** ones

“Maximal” means:

- if one resulting subsequence subsumes another, only the maximal one is kept
- the final sequence similarity is therefore a set of pairwise non-subsuming maximal common contiguous subsequences

## 8. Interpretation

This gives the following hierarchy:

- feature-level pattern structures
- product state pattern structure
- contiguous sequence pattern structure on state sequences

Numerical features are generalized by wider intervals.

Admissions are generalized by smaller threshold-like values.

Binary features are generalized by dropping presence constraints.

Sequence patterns are generalized by contiguous blocks of more general states.
