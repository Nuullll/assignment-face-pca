function [ projection_mat ] = kpca( V,D,K )
% keep k largest-magnitude valued eigenvectors
% V: eigen vectors in columns
% D: eigen values
% K: k

[~,I] = sort(D,'descend');
I = I(1:K);

projection_mat = V(:,I);

end

