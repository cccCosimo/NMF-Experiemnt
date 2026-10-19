% rng(1);
rng('shuffle');
m = 200; n = 300; r = 5; t = 12;
maxIter = 1000;

% Generate data with 5 percent noise
F0 = randn(m, r);
G0 = rand(n, r);
signal = F0 * G0.';
Z = randn(m, n);
X = signal + 0.05 * norm(signal, 'fro') / norm(Z, 'fro') * Z;

% Construct the compressed space
Omega = randn(n, t);
Y = X * Omega;
[Q, ~] = qr(Y, 0);
B = Q.' * X;

% The same initial value for G was used in both solutions
Ginit = max(0, randn(n, r));
[Ffull, Gfull, hfull] = seminmf_hals(X, Ginit, maxIter);
[Fs, Gs, hs] = seminmf_hals(B, Ginit, maxIter);

% Checking the compression and error identities
delta = norm(X - Q*B, 'fro');
e_small = norm(B - Fs*Gs.', 'fro');
e_full = norm(X - Q*Fs*Gs.', 'fro');

fprintf('Q orthogonality: %.3e\n', ...
    norm(Q.'*Q - eye(size(Q,2)), 'fro'));
fprintf('error identity: %.3e\n', ...
    abs(e_full^2 - delta^2 - e_small^2) / norm(X, 'fro')^2);

% Check HALS solutions error
fprintf('full-space relative error: %.6e\n', ...
    norm(X - Ffull*Gfull.', 'fro') / norm(X, 'fro'));
fprintf('sketched relative error:   %.6e\n', ...
    e_full / norm(X, 'fro'));
fprintf('minimum G entry: full %.3e, sketched %.3e\n', ...
    min(Gfull(:)), min(Gs(:)));

% Allowing floating-point arithmetic results in minimal fluctuations in the residuals
tol = 1e-10 * max([1, hfull(1), hs(1)]);
assert(all(diff(hfull) <= tol), 'Full-space objective increased.');
assert(all(diff(hs) <= tol), 'Sketched objective increased.');
assert(all(isfinite(hfull)) && all(isfinite(hs)), ...
    'Non-finite objective value.');
assert(all(isfinite(Gfull(:))) && all(isfinite(Gs(:))), ...
    'Non-finite entry in G.');
assert(all(Gfull(:) >= 0) && all(Gs(:) >= 0), ...
    'G contains a negative entry.');