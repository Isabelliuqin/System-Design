clear all;
close all;
clc


%fringe period lambda h

%%%read phi with object
I_1high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 1 f55.bmp');
[m n] = size(I_1high);
I_2high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 2 f55.bmp');
 
I_3high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 3 f55.bmp');

I_4high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 4 f55.bmp');

 
I_1high=mat2gray((I_1high), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2high=mat2gray((I_2high), [0 100000]);
I_3high=mat2gray((I_3high), [0 100000]);
I_4high=mat2gray((I_4high), [0 100000]);


%4 step algorithm
A_high=(I_4high - I_2high );
B_high=(I_1high - I_3high);

for i=1:m
     for j=1:n
         phi_high(i,j)= atan2(B_high(i,j),A_high(i,j));
             
     end
end
%}

%%%read phi without object
I_1_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 1 f55.bmp');

%automatic evaluation of wavelength of fringes projected
[m0 n0] = size(I_1_0high);

A = I_1_0high(30,:);
TF = islocalmax(A);

[r,c] = find(TF);

for i=1:45
    if i >= 2 & i<= 44
        c_shifted(i+1) = c(i);
        spacing(i) = c(i)-c_shifted(i);
    else
    end
end

%spacing(spacing<10) = 0;
%get mean of all spacing with value > 10
tf = spacing > 10;
Mean_spacing = mean(reshape(spacing(tf),1,[]));

lambda_high = Mean_spacing;

I_2_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 2 f55.bmp');
 
I_3_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 3 f55.bmp');

I_4_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 4 f55.bmp');

 
I_1_0high=mat2gray((I_1_0high), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2_0high=mat2gray((I_2_0high), [0 100000]);
I_3_0high=mat2gray((I_3_0high), [0 100000]);
I_4_0high=mat2gray((I_4_0high), [0 100000]);


%4 step algorithm
A0_high=(I_4_0high - I_2_0high );
B0_high=(I_1_0high - I_3_0high);

for i=1:m
     for j=1:n
         phi_0_high(i,j)= atan2(B0_high(i,j),A0_high(i,j));
         end
end

delta_phi_o_high = phi_high - phi_0_high;
delta_phi_o_high(isnan(delta_phi_o_high))=0; %set all NaN value in deltaphi =0

delta_phi_a_high = delta_phi_o_high;


%%%fringe period lambda l

%%%read phi with object
I_1low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 1 f50.bmp');

[m n] = size(I_1low);
I_2low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 2 f50.bmp');

I_3low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 3 f50.bmp');

I_4low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\fringe 4 f50.bmp');

 
I_1low=mat2gray((I_1low), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2low=mat2gray((I_2low), [0 100000]);
I_3low=mat2gray((I_3low), [0 100000]);
I_4low=mat2gray((I_4low), [0 100000]);


%4 step algorithm
Alow=(I_4low - I_2low );
Blow=(I_1low - I_3low);

for i=1:m
     for j=1:n
         phi_low(i,j)= atan2(Blow(i,j),Alow(i,j));
              
     end
end



%%%read phi without object
I_1_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 1 f50.bmp');
[m n] = size(I_1_0low);

%automatic evaluation of wavelength of fringes projected

B = I_1_0low(30,:);
TF_low = islocalmax(B);

[rl,cl] = find(TF_low);

for i=1:40
    if i >= 2 & i<= 39
        c_shifted_low(i+1) = cl(i);
        spacing_l(i) = cl(i)-c_shifted_low(i);
    else
    end
end

%get mean of all spacing with value > 10
tf_l = spacing_l > 10; 
Mean_spacing_l = mean(reshape(spacing_l(tf_l),1,[]));
lambda_low = Mean_spacing_l;

I_2_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 2 f50.bmp');

I_3_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 3 f50.bmp');

I_4_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\07022020\only fringe 4 f50.bmp');

 
I_1_0low=mat2gray((I_1_0low), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2_0low=mat2gray((I_2_0low), [0 100000]);
I_3_0low=mat2gray((I_3_0low), [0 100000]);
I_4_0low=mat2gray((I_4_0low), [0 100000]);


%4 step algorithm
A0low=(I_4_0low - I_2_0low );
B0low=(I_1_0low - I_3_0low);

for i=1:m
     for j=1:n
         phi_0_low(i,j)= atan2(B0low(i,j),A0low(i,j));
         end
end

delta_phi_o_low = phi_low - phi_0_low;
delta_phi_o_low(isnan(delta_phi_o_low))=0; %set all NaN value in deltaphi =0

delta_phi_a_low = delta_phi_o_low;


%%%multifrequency unwrapping
tic;
unwrap_img = unwrap_phase(phi_high);
toc;

%Unwrap reference phi
unwrap_img_0 = unwrap_phase(phi_0_high);
unwrap_img_0 = unwrap_phase(unwrap_img_0);


%figure(7)
imshow(unwrap_img,[])
%pcolor(unwrap_img)
shading flat;
set(gca, 'ydir', 'reverse');
title('Wrapped phase');


%figure(70)
imshow(unwrap_img_0,[])
%pcolor(unwrap_img)
shading flat;
set(gca, 'ydir', 'reverse');
title('Wrapped phase');

ave_col1 = mean(unwrap_img(:,1));
ave_col1_0 = mean(unwrap_img_0(:,1));

a = ave_col1;
Ave_col1 = a(ones(m, n));

b = ave_col1_0;
Ave_col1_0 = b(ones(m, n));

unwrap_img_before = unwrap_img - ave_col1;
unwrap_img_0 = unwrap_img_0 - ave_col1_0;


delta_phi = unwrap_img_before - unwrap_img_0;

unwrap_img_after1 = unwrap_phase(delta_phi);
unwrap_img_after2 = unwrap_phase(unwrap_img_after1);

unwrap_img_after3 = medfilt2(unwrap_img_after2,[10 10]);

delta_phi_eq = unwrap_img_after3;
%delta_phi_eq = delta_phi_a_high - delta_phi_a_low;
lambda_eq = (lambda_low * lambda_high)/(lambda_low - lambda_high);
k_h = round(((lambda_eq/lambda_high)*delta_phi_eq - delta_phi_a_high)/(2*pi));

delta_phi_unwrapped = delta_phi_a_high + 2*pi*k_h;

figure (1);mesh(delta_phi_unwrapped);
figure (2);imshow(delta_phi_unwrapped);

function res_img = unwrap_phase(img)
    [Ny, Nx] = size(img);

    % get the reliability
    reliability = get_reliability(img); % (Ny,Nx)

    % get the edges
    [h_edges, v_edges] = get_edges(reliability); % (Ny,Nx) and (Ny,Nx)

    % combine all edges and sort it
    edges = [h_edges(:); v_edges(:)];
    edge_bound_idx = Ny * Nx; % if i <= edge_bound_idx, it is h_edges
    [~, edge_sort_idx] = sort(edges, 'descend');

    % get the indices of pixels adjacent to the edges
    idxs1 = mod(edge_sort_idx - 1, edge_bound_idx) + 1;
    idxs2 = idxs1 + 1 + (Ny - 1) .* (edge_sort_idx <= edge_bound_idx);

    % label the group
    group = reshape([1:numel(img)], Ny*Nx, 1);
    is_grouped = zeros(Ny*Nx,1);
    group_members = cell(Ny*Nx,1);
    for i = 1:size(is_grouped,1)
        group_members{i} = i;
    end
    num_members_group = ones(Ny*Nx,1);

    % propagate the unwrapping
    res_img = img;
    num_nan = sum(isnan(edges)); % count how many nan-s and skip them
    for i = num_nan+1 : length(edge_sort_idx)
        % get the indices of the adjacent pixels
        idx1 = idxs1(i);
        idx2 = idxs2(i);

        % skip if they belong to the same group
        if (group(idx1) == group(idx2)) continue; end

        % idx1 should be ungrouped (swap if idx2 ungrouped and idx1 grouped)
        % otherwise, activate the flag all_grouped.
        % The group in idx1 must be smaller than in idx2. If initially
        % group(idx1) is larger than group(idx2), then swap it.
        all_grouped = 0;
        if is_grouped(idx1)
            if ~is_grouped(idx2)
                idxt = idx1;
                idx1 = idx2;
                idx2 = idxt;
            elseif num_members_group(group(idx1)) > num_members_group(group(idx2))
                idxt = idx1;
                idx1 = idx2;
                idx2 = idxt;
                all_grouped = 1;
            else
                all_grouped = 1;
            end
        end

        % calculate how much we should add to the idx1 and group
        dval = floor((res_img(idx2) - res_img(idx1) + pi) / (2*pi)) * 2*pi;

        % which pixel should be changed
        g1 = group(idx1);
        g2 = group(idx2);
        if all_grouped
            pix_idxs = group_members{g1};
        else
            pix_idxs = idx1;
        end

        % add the pixel value
        if dval ~= 0
            res_img(pix_idxs) = res_img(pix_idxs) + dval;
        end

        % change the group
        len_g1 = num_members_group(g1);
        len_g2 = num_members_group(g2);
        group_members{g2}(len_g2+1:len_g2+len_g1) = pix_idxs;
        group(pix_idxs) = g2; % assign the pixels to the new group
        num_members_group(g2) = num_members_group(g2) + len_g1;

        % mark idx1 and idx2 as already being grouped
        is_grouped(idx1) = 1;
        is_grouped(idx2) = 1;
    end
end

function rel = get_reliability(img)
    rel = zeros(size(img));

    % get the shifted images (N-2, N-2)
    img_im1_jm1 = img(1:end-2, 1:end-2);
    img_i_jm1   = img(2:end-1, 1:end-2);
    img_ip1_jm1 = img(3:end  , 1:end-2);
    img_im1_j   = img(1:end-2, 2:end-1);
    img_i_j     = img(2:end-1, 2:end-1);
    img_ip1_j   = img(3:end  , 2:end-1);
    img_im1_jp1 = img(1:end-2, 3:end  );
    img_i_jp1   = img(2:end-1, 3:end  );
    img_ip1_jp1 = img(3:end  , 3:end  );

    % calculate the difference
    gamma = @(x) sign(x) .* mod(abs(x), pi);
    H  = gamma(img_im1_j   - img_i_j) - gamma(img_i_j - img_ip1_j  );
    V  = gamma(img_i_jm1   - img_i_j) - gamma(img_i_j - img_i_jp1  );
    D1 = gamma(img_im1_jm1 - img_i_j) - gamma(img_i_j - img_ip1_jp1);
    D2 = gamma(img_im1_jp1 - img_i_j) - gamma(img_i_j - img_ip1_jm1);

    % calculate the second derivative
    D = sqrt(H.*H + V.*V + D1.*D1 + D2.*D2);

    % assign the reliability as 1 / D
    rel(2:end-1, 2:end-1) = 1./D;

    % assign all nan's in rel with non-nan in img to 0
    % also assign the nan's in img to nan
    rel(isnan(rel) & ~isnan(img)) = 0;
    rel(isnan(img)) = nan;
end

function [h_edges, v_edges] = get_edges(rel)
    [Ny, Nx] = size(rel);
    h_edges = [rel(1:end, 2:end) + rel(1:end, 1:end-1), nan(Ny, 1)];
    v_edges = [rel(2:end, 1:end) + rel(1:end-1, 1:end); nan(1, Nx)];
end







