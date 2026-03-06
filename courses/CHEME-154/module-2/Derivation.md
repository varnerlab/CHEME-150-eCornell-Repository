### Derivation of the logistic update probability

Starting from

$$
P\left(s_{i}^{(t)} = 1 \mid s_{\lnot i}\right)=
\frac{\exp\left(-\beta\,E\left(s_{i}=1\mid s_{\lnot i}\right)\right)}
{\exp\left(-\beta\,E\left(s_{i}=1\mid s_{\lnot i}\right)\right)+\exp\left(-\beta\,E\left(s_{i}=-1\mid s_{\lnot i}\right)\right)}
$$

For fixed $s_{\lnot i}$, the local conditional energy is

$$
E\left(s_i\mid s_{\lnot i}\right)=-s_i h_i^{(t)}
$$

so the two possible values are

$$
E\left(s_{i}=1\mid s_{\lnot i}\right)=-h_i^{(t)},
\qquad
E\left(s_{i}=-1\mid s_{\lnot i}\right)=+h_i^{(t)}.
$$

Substitute into the probability expression:

$$
\begin{align*}
P\left(s_{i}^{(t)} = 1 \mid s_{\lnot i}\right)
&= \frac{\exp\left(-\beta(-h_i^{(t)})\right)}{\exp\left(-\beta(-h_i^{(t)})\right)+\exp\left(-\beta(h_i^{(t)})\right)} \\
&= \frac{\exp\left(\beta h_i^{(t)}\right)}{\exp\left(\beta h_i^{(t)}\right)+\exp\left(-\beta h_i^{(t)}\right)} \\
&= \frac{1}{1+\exp\left(-2\beta h_i^{(t)}\right)}.
\end{align*}
$$

Thus,

$$
\boxed{
P\left(s_{i}^{(t)} = 1 \mid s_{\lnot i}\right)
= \frac{\exp\left(\beta h_i^{(t)}\right)}{\exp\left(\beta h_i^{(t)}\right)+\exp\left(-\beta h_i^{(t)}\right)}
= \frac{1}{1+\exp\left(-2\beta h_i^{(t)}\right)}
}\quad\blacksquare
$$