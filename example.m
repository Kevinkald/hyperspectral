clc; clear;
data = load('data/HICO.mat');
HICO_original = data.HICO_original;
HICO_noisy = data.HICO_noisy;
hico_wl = data.hico_wl;

land_mask         = double(imread('data/land_mask.png'));
land_mask         = land_mask(:,:,1);
deep_water_Rrs    = load('data/deep_water_Rrs.txt');
shallow_water_Rrs = load('data/shallow_water_Rrs.txt');
valid_bands_Rrs   = load('data/valid_bands_Rrs.txt');

size(HICO_original)
size(HICO_noisy)
size(hico_wl)
size(deep_water_Rrs)
size(shallow_water_Rrs)
size(valid_bands_Rrs)

% To convert a hyperspectral image cube I to matrix form X:
I = HICO_original;
[H,W,L] = size(I);
X = reshape(I, [H*W,L]);
X = X';

% To convert a matrix X back into a hyperspectral image cube:
I = reshape(X', [H,W,L]);

% To set all land pixels to zero:
% (See https://se.mathworks.com/matlabcentral/answers/38547-masking-out-image-area-using-binary-mask)
I = bsxfun(@times, I, cast(land_mask, 'like', I));

% Plot a single spectral band
imagesc(I(:,:,30));
axis('image');

% Note that quite a few libraries assume a matrix layout where
% each row is a spectral vector, rather than each column as in
% equation 2 of the assignment text. Read the documentation of
% those libraries carefully.
