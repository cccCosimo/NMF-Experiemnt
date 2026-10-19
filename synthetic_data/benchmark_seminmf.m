%BENCHMARK_SEMINMF Small synthetic comparison of full and sketched Semi-NMF.
% All reported errors are measured in the original space.

m = 200; n = 300; r = 5;
sketchSizes = [8, 12, 20, 40];
maxIter = 100;
seeds = 1:3;
noiseLevel = 0.05;

numSeeds = numel(seeds);
numSizes = numel(sketchSizes);
fullError = zeros(numSeeds, 1);
fullTime = zeros(numSeeds, 1);
projectionError = zeros(numSeeds, numSizes);
smallError = zeros(numSeeds, numSizes);
sketchedError = zeros(numSeeds, numSizes);
sketchTime = zeros(numSeeds, numSizes);
solveTime = zeros(numSeeds, numSizes);
recoveryTime = zeros(numSeeds, numSizes);
totalTime = zeros(numSeeds, numSizes);

for a = 1:numSeeds
    rng(seeds(a));
    F0 = randn(m, r);
    G0 = rand(n, r);
    signal = F0 * G0.';
    Z = randn(m, n);
    X = signal + noiseLevel * norm(signal, 'fro') / norm(Z, 'fro') * Z;
    Ginit = rand(n, r);
    OmegaMax = randn(n, max(sketchSizes));
    normX = norm(X, 'fro');

    % One full-data baseline for each seed, shared by all sketch sizes.
    [Ffull, Gfull] = seminmf_hals(X, Ginit, maxIter);
    fullError(a) = norm(X - Ffull*Gfull.', 'fro') / normX;
    fullTime(a) = timeit(@() seminmf_hals(X, Ginit, maxIter));

    for b = 1:numSizes
        t = sketchSizes(b);
        Omega = OmegaMax(:, 1:t);
        sketch = sketch_stage(X, Omega);
        Q = sketch.Q;
        B = sketch.B;
        [Fs, Gs] = seminmf_hals(B, Ginit, maxIter);
        Frecovered = recover_stage(Q, Fs);

        projectionError(a,b) = norm(X - Q*B, 'fro') / normX;
        smallError(a,b) = norm(B - Fs*Gs.', 'fro') / normX;
        sketchedError(a,b) = norm(X - Frecovered*Gs.', 'fro') / normX;

        % This identity checks that projection, compressed fit and recovery
        % were measured consistently; it does not certify solver convergence.
        identityGap = abs(sketchedError(a,b)^2 - ...
            projectionError(a,b)^2 - smallError(a,b)^2);
        assert(identityGap < 1e-10, 'Error identity failed.');

        % Each timeit repeats its complete function call. The four times
        % are measured separately and need not add up exactly.
        sketchTime(a,b) = timeit(@() sketch_stage(X, Omega));
        solveTime(a,b) = timeit(@() seminmf_hals(B, Ginit, maxIter));
        recoveryTime(a,b) = timeit(@() recover_stage(Q, Fs));
        totalTime(a,b) = timeit(@() sketched_pipeline( ...
            X, Omega, Ginit, maxIter));

        fprintf('seed %d, t %d: relative error %.4e, total %.4f s\n', ...
            seeds(a), t, sketchedError(a,b), totalTime(a,b));
    end
end

fprintf('\nAll runs use %d HALS iterations (fixed count).\n', maxIter);
fprintf('Full-data baseline: median error %.4e, median solve time %.4f s\n', ...
    median(fullError), median(fullTime));
fprintf('Ranges across seeds: error [%.4e, %.4e], time [%.4f, %.4f] s\n\n', ...
    min(fullError), max(fullError), min(fullTime), max(fullTime));

results = table(sketchSizes(:), median(projectionError,1).', ...
    median(smallError,1).', median(sketchedError,1).', ...
    min(sketchedError,[],1).', max(sketchedError,[],1).', ...
    median(sketchTime,1).', median(solveTime,1).', ...
    median(recoveryTime,1).', median(totalTime,1).', ...
    'VariableNames', {'t', 'projectionError', 'smallError', ...
    'fullSpaceError', 'errorMin', 'errorMax', 'sketchSeconds', ...
    'solveSeconds', 'recoverySeconds', 'totalSeconds'});
disp(results);

plot_seminmf_benchmark(sketchSizes, fullError, fullTime, ...
    projectionError, smallError, sketchedError, totalTime);

function sketch = sketch_stage(X, Omega)
    Y = X * Omega;
    [Q, ~] = qr(Y, 0);
    sketch.Q = Q;
    sketch.B = Q.' * X;
end

function F = recover_stage(Q, Fs)
    F = Q * Fs;
end

function F = sketched_pipeline(X, Omega, Ginit, maxIter)
    stage = sketch_stage(X, Omega);
    [Fs, ~] = seminmf_hals(stage.B, Ginit, maxIter);
    F = recover_stage(stage.Q, Fs);
end
