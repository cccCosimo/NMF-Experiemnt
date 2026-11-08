dataFile = '/Users/huangjiahao/CNMF_with_compression/Experiments/real_data/ionosphere/ionosphere/ionosphere.data';
fid = fopen(dataFile, 'r');
assert(fid ~= -1, 'Cannot open ionosphere.data');

format = [repmat('%f', 1, 34), '%s'];
raw = textscan(fid, format,'Delimiter', ',',...
    'CollectOutput', true);
fclose(fid);

X = raw{1}.';                 % 34 × 351：
classText = raw{2};           % 351 × 1：'g' 或 'b'
labels = zeros(numel(classText), 1);
labels(strcmp(classText, 'g')) = 1;
labels(strcmp(classText, 'b')) = 2;

assert(isequal(size(X), [34, 351]));
assert(all(isfinite(X(:))));
assert(all(labels == 1 | labels == 2));

fprintf('X: %d × %d, range: [%.3f, %.3f]\n', ...
    size(X, 1), size(X, 2), min(X(:)), max(X(:)));
fprintf('g: %d, b: %d\n', nnz(labels == 1), nnz(labels == 2));