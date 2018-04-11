% face PCA

%% input directory
TRAIN_DIR = '../train/';
TEST_DIR = '../test/';

K_MAX = 361;

%% train set PCA
D = dir([TRAIN_DIR,'*.jpg']);

count = length(D);

A = [];     % data

images = {D.name};

for i=1:count
    name = images{i};
    
    img = imread([TRAIN_DIR,name]);
    
    A(:,end+1) = double(img(:));
end

% covariance matrix
mu = mean(A,2);     % mean face
C = cov(A');

% eigen vectors
[V,D] = eig(C,'vector');

%% prepare test data
% (a) pick a random image in the train set
name = images{randi(count,1)};
xa = imread([TRAIN_DIR,name]);
xa = double(xa(:));
% (b) face.jpg
xb = imread([TEST_DIR,'face.jpg']);
xb = double(xb(:));
% (c) nonface.jpg
xc = imread([TEST_DIR,'nonface.jpg']);
xc = double(xc(:));

x = [xa,xb,xc];

%% reconstruction
% loop through K
errors = [];
for K = 1:K_MAX
    % keep k eigenvectors and get the projection matrix
    W = kpca(V,D,K);

    x_r = mu + W*W'*(x-mu);

    error = sum((x-x_r).^2);
    errors(end+1,:) = error;
end

%% plot error-K curve
plot(1:K_MAX,errors);
xlabel('K');
ylabel('error');
legend([name,' (train set)'],'face.jpg (test set)','nonface.jpg (test set)');