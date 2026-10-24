r = 40;
t = 240;
maxIter = 150;
normX = norm(X, 'fro');

% Uses the same initial G as the full-data experiment
rng(1);
Ginit = rand(size(X,2), r);

% Repeat the random projection at t=160
rng(11);
OmegaMax = randn(size(X,2), 240);
Omega = OmegaMax(:, 1:t);

totalStart = tic;

stageStart = tic;
Y = X * Omega;
[Q, ~] = qr(Y, 0);
B = Q.' * X;
sketchSeconds = toc(stageStart);

stageStart = tic;
[Fs, Gs, historyS] = seminmf_hals(B, Ginit, maxIter);
solveSeconds = toc(stageStart);

stageStart = tic;
Fsketch = Q * Fs;
recoverySeconds = toc(stageStart);

factorizationSeconds = toc(totalStart);

projectionError = norm(X - Q*B, 'fro') / normX;
smallError = norm(B - Fs*Gs.', 'fro') / normX;
fullError = norm(X - Fsketch*Gs.', 'fro') / normX;
identityGap = abs(fullError^2 - projectionError^2 - smallError^2);

% The same clustering rules and k-means settings were used as in the full-data experiment
Zs = Gs .* vecnorm(Fsketch, 2, 1);
Zs = Zs ./ max(vecnorm(Zs, 2, 2), eps);
rng(2);
stageStart = tic;
clusterSketch = kmeans(Zs, 40, ...
    'Replicates', 20, 'MaxIter', 300, ...
    'EmptyAction', 'singleton');
kmSeconds = toc(stageStart);

Csketch = accumarray([labels, clusterSketch], 1, [40, 40]);
puritySketch = sum(max(Csketch, [], 1)) / numel(labels);

fprintf('Projection/Small spaces/Original space error：%.6f / %.6f / %.6f\n', ...
    projectionError, smallError, fullError);
fprintf('Error identity deviation：%.2e\n', identityGap);
fprintf('Compress/Solve/Restore/Total time：%.2f / %.2f / %.2f / %.2f s\n', ...
    sketchSeconds, solveSeconds, recoverySeconds, factorizationSeconds);
fprintf('k-means solve time: %.4f s\n', kmSeconds);
fprintf('compressed Purity：%.4f\n', puritySketch);

assert(identityGap < 1e-10);