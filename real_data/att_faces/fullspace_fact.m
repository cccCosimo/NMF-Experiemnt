addpath('/Users/huangjiahao/CNMF_with_compression/Experiments/synthetic_data');

r = 40;     %objective dimention of factorization
maxIter = 50;
rng(1);
Ginit = rand(size(X, 2), r);

% compute factorization
tic;
[Ffull, Gfull, history] = seminmf_hals(X, Ginit, maxIter);
solveSeconds = toc;

relativeError = norm(X - Ffull * Gfull.', 'fro') / norm(X, 'fro');
relativeCurve = sqrt(2 * history) / norm(X, 'fro');

fprintf('solve time: %.2f s\n', solveSeconds);
fprintf('relative error in original space: %.6f\n', relativeError);
fprintf('Gs minimul element: %.3e\n', min(Gfull(:)));
fprintf('original/final value of objective function: %.6e / %.6e\n', history(1), history(end));

% assert if something went wrong
assert(isequal(size(Ffull), [size(X,1), r]));
assert(isequal(size(Gfull), [size(X,2), r]));
assert(all(isfinite(history)));
assert(all(Gfull(:) >= 0));
assert(abs(relativeCurve(end) - relativeError) < 1e-10);

figure
plot(0:maxIter, relativeCurve, 'LineWidth', 1.5);
xlabel('HALS iteration number');
ylabel('relative reconstruct erros in original space');
grid on;


