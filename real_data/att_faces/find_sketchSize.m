sketchSizes = [80, 120, 160, 240];
rng(11);
OmegaMax = randn(size(X,2), max(sketchSizes));

normX = norm(X, 'fro');
projectionError = zeros(size(sketchSizes));
sketchSeconds = zeros(size(sketchSizes));

for j = 1:numel(sketchSizes)
    t = sketchSizes(j);
    tic;
    Y = X * OmegaMax(:, 1:t);
    [Q, ~] = qr(Y, 0);
    B = Q.' * X;
    sketchSeconds(j) = toc;

    projectionError(j) = norm(X - Q*B, 'fro') / normX;

    fprintf('t=%3d  projective error=%.6f  compressing time=%.2f s  orthogonal error=%.2e\n', ...
        t, projectionError(j), sketchSeconds(j), ...
        norm(Q.'*Q - eye(t), 'fro'));
end