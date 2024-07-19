# MQE
## This repository is the official implementation of "Noise-Resilient Unsupervised Graph Representation Learning via Multi-Hop Feature Quality Estimation", accepted by CIKM 2024 


\begin{algorithm}[t]
\caption{The training algorithm of \ourmethod}
\label{alg:overall}
\LinesNumbered
\KwIn{Number of epoch $T$; Feature matrix $\mathbf{X}$; Adjacency matrix $\mathbf{A}$; Propagation iteration: $L$; Top-K neighbors $K$}
\KwOut{representations $\mathbf{Z}$; Propagated features quality $\sigma$}
\tcc{Augmented Message Passing}
% Initialize $\mathbf{X}^0=\mathbf{X}, \tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}, \hat{\mathbf{A}}=\widetilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \widetilde{\mathbf{D}}^{-\frac{1}{2}}$\\
Calculate $\mathbf{X}^*, \mathbf{A}^s \longleftarrow Prop(\mathbf{X},\mathbf{A};L)$ via Eq.~\eqref{eq.4-1}, ~\eqref{eq.4-2}\\
Calculate $\mathbf{A}^* \longleftarrow kNN(\mathbf{X}^*;K)$ via Eq.~\eqref{eq.4-3}\\
Calculate $\hat{\mathbf{X}} \longleftarrow Prop(\mathbf{X}, \mathbf{A}^*;L)$ via Eq.~\eqref{eq.4-3} \\ 
\tcc{Features Quality Estimation}
Initialize model parameters; Meta representations $\mathbf{Z}$\\
% \While{not convergence}{
\For{$t$=1:$T$}{
    \For{$\ell$ = 1:$L$}{
    \For{$i$ = 1:$n$}{
        Calculate $p\left(\hat{\mathbf{x}}_i^{(\ell)} \mid \mathbf{z}_i\right) \longleftarrow E(\mathbf{z}_i)$ via Eq.~\eqref{eq.4-5}
    }
    }
    Calculate loss $\mathcal{L}$ via Eq.~\eqref{eq.4-7}\\
    Update $E$ and $\mathbf{Z}$ via gradient descent\\
    }
% \vspace{-3mm}
\end{algorithm}
% \vspace{-3mm}
