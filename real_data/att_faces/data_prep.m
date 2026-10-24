dataRoot = fullfile(pwd, 'att_faces_data');

height = 112;
width = 92;           % 112 * 92 pixels per image
nPeople = 40;
nPerPerson = 10;
n = nPeople * nPerPerson;    % 10 images per person

% vectorize image 112 * 92 matrix
% each column of X represents a image
X = zeros(height * width, n);
labels = zeros(n, 1);

%load faces data
for person = 1:nPeople
    for imageNo = 1:nPerPerson
        col = (person - 1) * nPerPerson + imageNo;
        file = fullfile(dataRoot, sprintf('s%d', person),...
                        sprintf('%d.pgm', imageNo));
        
        img = imread(file);
        assert(isequal(size(img), [height, width]), ...
            'image does not conform to the formal size', file);
        
        X(:, col) = double(img(:)) / 255;
        labels(col) = person;
    end
end
% assert if somethiing went wrong
assert(isequal(size(X), [height*width, nPeople*nPerPerson]));
assert(all(isfinite(X(:))));
assert(all(X(:) >= 0 & X(:) <= 1));
assert(numel(unique(labels)) == 40);

fprintf('X: %d * %d\n', size(X, 1), size(X,2));
fprintf('pixel scale: [%.3f, %.3f]\n', min(X(:)), max(X(:)));

% diaplay few samples
figure;
for j = 1:10
    subplot(2, 5, j);
    imagesc(reshape(X(:, j), height, width));
    colormap gray;
    axis image off;
    title(sprintf('Person %d, image %d', labels(j), j));
end














