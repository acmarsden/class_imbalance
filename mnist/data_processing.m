% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

% We are using display_network from the autoencoder code
%k = 20;
%display_network(images(:,1:k)); % Show the first k images

% Pick two integers to distinguish between
i = 0; j = 1;
n_pos = sum(labels == i); n_neg = sum(labels == j);
x_pos = images(:,labels == i); x_neg = images(:,labels == j);
x = [x_pos, x_neg]; y = [ones(1, n_pos), -ones(1, n_neg)];
