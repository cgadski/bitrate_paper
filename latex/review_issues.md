# Review: Typos and Math Issues

## Section 1 (Introduction)

- [ ] 1. **Missing word "a"**: "If each neuron in given neural network" → "in **a** given neural network"

- [ ] 2. **Wrong variable in SAE estimate**: "$\hat x(y) = \sigma(G x)$" should be "$\hat x(y) = \sigma(G y)$" — the estimate takes $y$ as input, not $x$.

- [ ] 3. **Colon vs. "in"**: "$G \colon \R^{N \times d}$" — a colon here implies a function signature, but $\R^{N \times d}$ is a set of matrices. Should be "$G \in \R^{N \times d}$".

- [ ] 4. **"an accurate predictions"**: "leads to an accurate predictions" → "leads to accurate predictions"

- [ ] 5. **Extra "to"** (Figure 2 caption): "two different methods to were empirically found" → "two different methods were empirically found"

- [ ] 6. **Missing "a"** (Figure 2 caption): "GMP is simple algorithm" → "GMP is **a** simple algorithm"

- [ ] 7. **Typo "probabiliy"** (Figure 2 caption): "with probabiliy $0.95$" → "with probability $0.95$"

## Section 2 (Codes)

- [ ] 8. **Missing "a"**: "a linear projection of high-dimensional and sparse vector $x$" → "of **a** high-dimensional..."

- [ ] 9. **Missing "as"**: "viewing $F$ the matrix of column vectors" → "viewing $F$ **as** the matrix..."

- [ ] 10. **Missing "is"**: "once the support of $x$ identified" → "once the support of $x$ **is** identified"

- [ ] 11. **Redundant/contradictory quantifiers in Remark 1**: The remark says "Let $x \subseteq [N]$ be a $k$-sparse subset" then immediately "for any $k$-sparse vector $x \in \R^N$." The first sentence is redundant — just the universal quantifier is needed.

- [ ] 12. **Inconsistent notation for variance**: "Plugging this expression in as the variance $\Var\, \langle \lambda_i, Z \rangle$" — $\lambda_i$ was defined as a *function*, not a vector. This should be "$\Var\, \lambda_i(Z_i)$" to match the earlier definition.

- [ ] 13. **Typo "fictitous"**: "in our fictitous Gaussian model" → "fictitious"

## Section 3 (Information Theory)

- [ ] 14. **Nonstandard notation $\ln_2$**: "$\frac{1}{16} \ln_2 \binom{N}{k}$" — should be $\log_2$ (used correctly one line earlier).

- [ ] 15. **Probability > 1**: "if there is any pair $(F, G)$ that transmits $X$ with probability greater than $1 + o(1)$" — probability cannot exceed 1. Likely should be "with probability $1 - o(1)$" or "with probability bounded away from $0$".

- [ ] 16. **Missing "be"**: "can separated into two pieces" → "can **be** separated into"

- [ ] 17. **Missing "be"**: "information that would required to store" → "would **be** required to store"

## Section 4 (Main Results)

- [ ] 18. **Matrix dimensions transposed**: "$F_{N,d} \in \R^{N \times d}$" — throughout the paper $F$ maps $\R^N \to \R^d$, so $F \in \R^{d \times N}$.

- [ ] 19. **Missing square in limit expression**: The text says "its limit $2(1 + \sqrt\eta)/(1 - \eta)$ for large $N$ is indicated by a bold line." This should be $2(1 + \sqrt\eta)^2/(1 - \eta)$. (The figure caption correctly has the equivalent form $(2\ln 2)(1+\sqrt\eta)/(1-\sqrt\eta)$ in bits; the text in nats is missing the square.)

- [ ] 20. **Bold/thin/dotted line inconsistency between text and Figure 3 caption**: The text says the finite-$N$ expression is "dotted" and the limit is "bold." The caption says the finite-$N$ expression is "bold" and the limit is "thin." These should agree.

- [ ] 21. **Typo "occuring"** (Figure 3 caption): → "occurring"

## Section 5 (Dictionaries)

- [ ] 22. **Incorrect inequality in Welch bound derivation**: The text says the bound follows by "applying the inequality $\sum_i \lambda_i^2 \ge N \sum_i \lambda_i$." This is wrong — it would give $\sum_i \lambda_i^2 \ge N^2$, which is far too strong. The correct justification is Cauchy–Schwarz on the (at most $d$) nonzero eigenvalues: $\sum_i \lambda_i^2 \ge (\sum_i \lambda_i)^2 / d = N^2/d$.

## Section 6 (Conclusions)

- [ ] 23. **Missing "models"**: "large language can be adequately modeled" → "large language **models** can be"

- [ ] 24. **Missing "more"**: "using slightly computation" → "using slightly **more** computation"

## Appendix: Proofs

- [ ] 25. **Typo $Y_i$ for $F_i$**: In the proof of Proposition 4 (necessary condition): "the pair $(Y_i, Y/\lVert Y \rVert)$ is a pair of independent, uniform draws from the unit sphere" — $Y_i$ is undefined; should be $F_i$.

- [ ] 26. **Undefined label `\Cref{eq:tau-map}`**: The appendix references `\Cref{eq:tau-map}`, but no equation with this label exists. The $\tau_\text{MAP}$ expression in Section 2 is not labeled (the nearby labeled equation is `eq:tau`).

- [ ] 27. **Missing period**: "how it relates to the MAP threshold derived in \Cref{sec:codes}" at the end of a paragraph — missing terminal period.