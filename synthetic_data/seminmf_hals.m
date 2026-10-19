function [F, G, history] = seminmf_hals(D, Ginit, maxIter)
%SEMINMF_HALS using the least squares method and column-by-column HALS to
%solve constrained least square:
%   min_{F, G >= 0} 0.5 * ||D - F*G.'||_F^2
%   history(k+1) is the value of the objective function at the end of the kth iteration.

    G = Ginit;
    F = lsqminnorm(G, D.').';

    history = zeros(maxIter + 1, 1);
    history(1) = 0.5 * norm(D - F*G.', 'fro')^2;

    for k = 1:maxIter
        C = D.' * F;
        S = F.' * F;

        for j = 1:size(G, 2)
            if S(j,j) > 0
                d = C(:,j) - G*S(:,j);
                G(:,j) = max(0, G(:,j) + d/S(j,j));
            end
        end

        % Pair F with G, as updated in this round
        F = lsqminnorm(G, D.').';
        history(k+1) = 0.5 * norm(D - F*G.', 'fro')^2;
    end
end