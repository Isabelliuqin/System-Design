clear all;
close all;
clc



%%%read phi with object
I_1= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 9 2.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1);
I_2= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 10 2.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 11 2.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 12 2.bmp');

 
I_1=mat2gray((I_1), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2=mat2gray((I_2), [0 100000]);
I_3=mat2gray((I_3), [0 100000]);
I_4=mat2gray((I_4), [0 100000]);

%4 step algorithm
A=(I_4 - I_2 );
B=(I_1 - I_3);

for i=1:m
     for j=1:n
         phi(i,j)= atan2(B(i,j),A(i,j));
              %pause(0.3)
     end
end

%%%read phi without object
I_1_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 9 2.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1);
I_2_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 10 2.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 11 2.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 12 2.bmp');

 
I_1_0=mat2gray((I_1_0), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2_0=mat2gray((I_2_0), [0 100000]);
I_3_0=mat2gray((I_3_0), [0 100000]);
I_4_0=mat2gray((I_4_0), [0 100000]);


%4 step algorithm
A0=(I_4_0 - I_2_0 );
B0=(I_1_0 - I_3_0);

for i=1:m
     for j=1:n
         phi_0(i,j)= atan2(B0(i,j),A0(i,j));
         end
end

%ave_col1 = mean(phi(:,1));
%ave_col1_0 = mean(phi_0(:,1));

%a = ave_col1;
%Ave_col1 = a(ones(m, n));


%phi_corrected = phi - ave_col1;          
%b = ave_col1_0;
%Ave_col1_0 = b(ones(m, n));   %place them at the same reference level
%phi_corrected_0 = phi_0 - ave_col1_0;

delta_phi_o = phi - phi_0;
delta_phi_o(isnan(delta_phi_o))=0; %set all NaN value in deltaphi =0

delta_phi_a = delta_phi_o;



%%%noise prefilter
alpha = pi/2;
beta = 1;
gamma = 0.5;
windowsize = 9;


for i = 400:512
    for j = 400:512
        window = delta_phi_o(i-4:i+4,j-4:j+4);
        H1 = [];
        H2 = [];
        H3 = [];
        for p = 1:windowsize
            for q = 1:windowsize
                if p == 1 & q ~= 1 & q ~= windowsize
                    if abs(window(p,q+1) - window(p,q)) > pi & abs(window(p,q) - window(p,q-1)) > pi
                        window(p,q) = 0;
                    else
                    end
                elseif p == windowsize & q ~= 1 & q ~= windowsize
                     if abs(window(p,q+1) - window(p,q)) > pi & abs(window(p,q) - window(p,q-1)) > pi
                        window(p,q) = 0;
                    else
                     end
                
                elseif q ~= 1 & q ~= windowsize
                    if abs(window(p+1,q) - window(p,q)) > pi & abs(window(p,q) - window(p-1,q)) > pi
                        window(p,q) = 0;
                    else
                    end
                else
                end
                
                if window(p,q) < -alpha
                    H1 = [H1 window(p,q)];
                    
                elseif window(p,q) >= -alpha & window(p,q) <= alpha & window(p,q)~=0
                    H2 = [H2 window(p,q)];
                
                elseif window(p,q) > alpha
                    H3 = [H3 window(p,q)];
                else
                 
                end
            end
        end
        
        G = length(H1) + length(H2) + length(H3);
        if isempty(H1)
            H1(1) = 0;
        end
        if isempty(H2)
            H2(1) = 0;
        end
        if isempty(H3)
            H3(1) = 0;
        end
        if length(H2) > beta*(length(H1)+length(H3))
            delta_phi_a(i,j) = median([H1 H2 H3]);
        elseif length(H2)<= beta*(length(H1)+length(H3)) & length(H3) > length(H1) & length(H3) < gamma*G & length(H1) < length(H2)
            delta_phi_a(i,j) = median([H2 H3]);
        elseif length(H2)<= beta*(length(H1)+length(H3)) & length(H1) >= length(H3) & length(H1) < gamma*G & length(H3) < length(H2)
            delta_phi_a(i,j) = median([H1 H2]);
        elseif length(H2)<= beta*(length(H1)+length(H3)) & length(H3) > length(H1) & length(H3)>= gamma*G
            delta_phi_a(i,j) = median(H3);
        elseif length(H2)<= beta*(length(H1)+length(H3)) & length(H3) > length(H1) & length(H1)>=length(H2)
            delta_phi_a(i,j) = median(H3);
        elseif length(H2)<= beta*(length(H1)+length(H3)) & length(H1) >= length(H3) & length(H1)<= gamma*G
            delta_phi_a(i,j) = median(H1);
        elseif length(H2)<= beta*(length(H1)+length(H3)) & length(H1) >= length(H3) & length(H3)>= length(H2)
            delta_phi_a(i,j) = median(H1);
        end
    end
end

delta_phi_a(isnan(delta_phi_a))=0; %set all NaN value in deltaphi =0
filtereffect = delta_phi_a - delta_phi_o;
                    

%{
%manual noise treatment
[x4 y4]=find(delta_phi<-5);
for i = 1:length(x4)
    A = delta_phi(x4(i,1)-1:x4(i,1),y4(i,1)-2:y4(i,1)+2);
    delta_phi(x4(i),y4(i)) = (sum(sum(A),2)-delta_phi(x4(i),y4(i)))/9;
    %delta_phi(x4(i),y4(i)) = 0;
end
%}
unwrap_img_after1 = unwrap_phase(delta_phi_a);
unwrap_img_after2 = unwrap_phase(unwrap_img_after1);

unwrap_img_after3 = medfilt2(unwrap_img_after2,[10 10]);

figure(10)
imshow(unwrap_img_after3,[])

figure(20),mesh(unwrap_img_after3)

function res_img = unwrap_phase(img)
    [Ny, Nx] = size(img);

    % get the reliability
    reliability = get_reliability(img); % (Ny,Nx)

    % get the edges
    [h_edges, v_edges] = get_edges(reliability); % (Ny,Nx) and (Ny,Nx)

    % combine all edges and sort it
    edges = [h_edges(:); v_edges(:)]; %form a matrix of reliability of edges
    edge_bound_idx = Ny * Nx; % if i <= edge_bound_idx, it is h_edges
    [~, edge_sort_idx] = sort(edges, 'descend');  %sort reliability of edges from the largest to the lowest

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
    img_im1_jm1 = img(1:end-2, 1:end-2); %phi(i-1,j-1)
    img_i_jm1   = img(2:end-1, 1:end-2); %phi(i,j-1)
    img_ip1_jm1 = img(3:end  , 1:end-2); %phi(i+2,j-1)
    img_im1_j   = img(1:end-2, 2:end-1); %phi(i-1,j)
    img_i_j     = img(2:end-1, 2:end-1); %phi(i,j)
    img_ip1_j   = img(3:end  , 2:end-1); %phi(i+1,j)
    img_im1_jp1 = img(1:end-2, 3:end  ); %phi(i-1,j+1)
    img_i_jp1   = img(2:end-1, 3:end  ); %phi(i,j+1)
    img_ip1_jp1 = img(3:end  , 3:end  ); %phi(i+1.j+1)

    % calculate the difference
    gamma = @(x) sign(x) .* mod(abs(x), pi);
    H  = gamma(img_im1_j   - img_i_j) - gamma(img_i_j - img_ip1_j  );%a matrix containing all H for each pixel(i,j)
    V  = gamma(img_i_jm1   - img_i_j) - gamma(img_i_j - img_i_jp1  );
    D1 = gamma(img_im1_jm1 - img_i_j) - gamma(img_i_j - img_ip1_jp1);
    D2 = gamma(img_im1_jp1 - img_i_j) - gamma(img_i_j - img_ip1_jm1);

    % calculate the second derivative
    D = sqrt(H.*H + V.*V + D1.*D1 + D2.*D2);

    % assign the reliability as 1 / D
    rel(2:end-1, 2:end-1) = 1./D; %reliability for all pixels are calculated

    % assign all nan's in rel with non-nan in img to 0
    % also assign the nan's in img to nan
    rel(isnan(rel) & ~isnan(img)) = 0;
    rel(isnan(img)) = nan;
end

function [h_edges, v_edges] = get_edges(rel) %get reliability of horizontal edges and vertical edges 
    [Ny, Nx] = size(rel);
    h_edges = [rel(1:end, 2:end) + rel(1:end, 1:end-1), nan(Ny, 1)];  %a matrix calculated reliability of horizontal edges for all pixel edges
    v_edges = [rel(2:end, 1:end) + rel(1:end-1, 1:end); nan(1, Nx)];
end


