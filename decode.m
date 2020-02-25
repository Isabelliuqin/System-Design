
% Algoritmos para Phase Shifting

clear all;
close all;
clc


%shift needed to let all min fringe intensity at approx 0
tic
I_1_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 1.bmp');

A = I_1_0(30,:);
TF = islocalmin(A);
[r,c] = find(TF);
min_value = A(TF);

tf = min_value < 80; 
min_value_reshape = min_value(tf);
Mean_lift = mean(reshape(min_value_reshape,1,[]));


%%%read phi with object
I_1= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\wire\fringe 1.bmp');

[m n] = size(I_1);
I_2= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\wire\fringe 2.bmp');

 
I_3= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\wire\fringe 3.bmp');

I_4= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\wire\fringe 4.bmp');

I_5= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\wire\fringe 5.bmp');

I_6= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\wire\fringe 6.bmp');

 
I_1=mat2gray((I_1), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2=mat2gray((I_2), [0 100000]);
I_3=mat2gray((I_3), [0 100000]);
I_4=mat2gray((I_4), [0 100000]);
I_5=mat2gray((I_5), [0 100000]);
I_6=mat2gray((I_6), [0 100000]);



%4 step algorithm
A=(I_4 - I_2 );

B=(I_1 - I_3);


for i=1:m
     for j=1:n
         phi(i,j)= atan2(B(i,j),A(i,j));
         
     end
end


%{
%6 step algorithm

%2

A = -sqrt(3)*(I_2 + I_3 - I_5 - I_6);
B = 2*I_1 + I_2 - I_3 - 2*I_4 - I_5 + I_6;

for i=1:m
     for j=1:n
         phi(i,j)= atan2(B(i,j),A(i,j));
         
     end
end
%}

phi = phi - Mean_lift(ones(m,n));


%%%read phi without object
I_1_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 1.bmp');

[m n] = size(I_1);
I_2_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 2.bmp');

 
I_3_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 3.bmp');

I_4_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 4.bmp');

I_5_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 5.bmp');

I_6_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 6.bmp');

 
I_1_0=mat2gray((I_1_0), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2_0=mat2gray((I_2_0), [0 100000]);
I_3_0=mat2gray((I_3_0), [0 100000]);
I_4_0=mat2gray((I_4_0), [0 100000]);
I_5_0=mat2gray((I_5_0), [0 100000]);
I_6_0=mat2gray((I_6_0), [0 100000]);


%4 step algorithm
A0=(I_4_0 - I_2_0 );
B0=(I_1_0 - I_3_0);

 for i=1:m
     for j=1:n
         phi_0(i,j)= atan2(B0(i,j),A0(i,j));

     end
 end
 

%{
%6 step algorithm

A0 = -sqrt(3)*(I_2_0 + I_3_0 - I_5_0 - I_6_0);
B0 = 2*I_1_0 + I_2_0 - I_3_0 - 2*I_4_0 - I_5_0 + I_6_0;

for i=1:m
     for j=1:n
         phi_0(i,j)= atan2(B0(i,j),A0(i,j));
         
     end
end
%}
phi_0 = phi_0 - Mean_lift(ones(m,n));

%%%UNWRAPPING METHODS



%%%Noise reduction Filter

%%%%%%%%%%%%%%%%%%%%%%%%%%%Unwrapping Method 3%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%method


%unwrap phi phi_0 indivitually
unwrap_img_0 = unwrap_phase(phi_0);
unwrap_img = unwrap_phase(phi);



figure(7)

imshow(unwrap_img,[])
pcolor(unwrap_img)
shading flat;
set(gca, 'ydir', 'reverse');
title('Wrapped phase');


figure(70)
imshow(unwrap_img_0,[])
pcolor(unwrap_img)
shading flat;
set(gca, 'ydir', 'reverse');
title('Wrapped phase');


delta_phi = unwrap_img - unwrap_img_0;

%Filter object deltaphi
delta_phi_o = delta_phi;
delta_phi_o(isnan(delta_phi_o))=0; %set all NaN value in deltaphi =0

delta_phi_a = delta_phi_o;

%%%noise prefilter
alpha = pi/2;
beta = 1;
gamma = 0.5;
windowsize = 9;


for i = 5:1018
    for j = 5:1275
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
                    if abs(window(p,q+1) - window(p,q)) > pi & abs(window(p,q) - window(p,q-1)) > pi
                        window(p,q) = 0;
                    elseif abs(window(p+1,q) - window(p)) > pi & abs(window(p,q) - window(p-1,q)) > pi
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
filtereffect_0 = delta_phi_a - delta_phi_o;

unwrap_img_after1 = unwrap_phase(delta_phi_a);
unwrap_img_after2 = unwrap_phase(unwrap_img_after1);

unwrap_img_after3 = medfilt2(unwrap_img_after2,[10 10]);
unwrap_img_after4 = unwrap_phase(unwrap_img_after3);

%place the reference plane to zero
B = unwrap_img_after4(30,:);
TFb = islocalmin(B);
min_valueb = B(TFb);

tfb = min_valueb < 45; 
min_value_reshapeb = min_valueb(tfb);
Mean_liftb = mean(reshape(min_value_reshapeb,1,[]));

unwrap_img_after5 = unwrap_img_after4 - Mean_liftb;

figure(66)
imshow(unwrap_img_after5,[]);
figure(67);
mesh(unwrap_img_after5);

%height calibration
cali_factor = 2.9/0.5994;
caliheight = cali_factor * unwrap_img_after5;




toc

%max(max(caliheight))


%average height

E = caliheight(1024,:);
E(E>1)=0;
caliheight(1024,:)=E;

F = caliheight(:,1280);
F(F>1)=0;
caliheight(:,1280) = F;

figure(68);
%mesh(caliheight);
[x,y] = meshgrid(0:52/732:1279*52/732,0:52/732:1023*52/732);
mesh(x,y,caliheight);
ax = gca;
ax.FontSize = 25;
xlabel('x(mm)','FontSize',25);
ylabel('y(mm)','FontSize',25);
zlabel('height(mm)','FontSize',25);

h_array = caliheight(caliheight>0.1);
h_ave = mean(h_array,'all');



%{
%calibration
%shift needed to let all min fringe intensity at approx 0
tic
I_1_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 1.bmp');

A = I_1_0(30,:);
TF = islocalmin(A);
[r,c] = find(TF);
min_value = A(TF);

tf = min_value < 80; 
min_value_reshape = min_value(tf);
Mean_lift = mean(reshape(min_value_reshape,1,[]));


%%%read phi with object
I_1= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\calibration coin\fringe 1.bmp');

[m n] = size(I_1);
I_2= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\calibration coin\fringe 2.bmp');

 
I_3= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\calibration coin\fringe 3.bmp');

I_4= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\calibration coin\fringe 4.bmp');
I_5 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\calibration coin\fringe 5.bmp');
I_6 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\calibration coin\fringe 6.bmp');
 
I_1=mat2gray((I_1), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2=mat2gray((I_2), [0 100000]);
I_3=mat2gray((I_3), [0 100000]);
I_4=mat2gray((I_4), [0 100000]);
I_5=mat2gray((I_5), [0 100000]);
I_6=mat2gray((I_6), [0 100000]);

%{
%4 step algorithm
A=(I_4 - I_2 );

B=(I_1 - I_3);

 for i=1:m
     for j=1:n
         phi(i,j)= atan2(B(i,j),A(i,j));
         
     end
 end
%}

%6 step algorithm


A = -sqrt(3)*(I_2 + I_3 - I_5 - I_6);
B = 2*I_1 + I_2 - I_3 - 2*I_4 - I_5 + I_6;

for i=1:m
     for j=1:n
         phi(i,j)= atan2(B(i,j),A(i,j));
         
     end
end

 
 
phi = phi - Mean_lift(ones(m,n));


%%%read phi without object
I_1_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 1.bmp');

[m n] = size(I_1);
I_2_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 2.bmp');

 
I_3_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 3.bmp');

I_4_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 4.bmp');
I_5_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 5.bmp');

I_6_0= imread('D:\IC\Master degree\Laboratory\System Design\experiment\24022020\only\only fringe 6.bmp');

 
I_1_0=mat2gray((I_1_0), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2_0=mat2gray((I_2_0), [0 100000]);
I_3_0=mat2gray((I_3_0), [0 100000]);
I_4_0=mat2gray((I_4_0), [0 100000]);
I_5_0=mat2gray((I_5_0), [0 100000]);
I_6_0=mat2gray((I_6_0), [0 100000]);

%{
%4 step algorithm
A0=(I_4_0 - I_2_0 );
B0=(I_1_0 - I_3_0);

 for i=1:m
     for j=1:n
         phi_0(i,j)= atan2(B0(i,j),A0(i,j));

     end
 end
%}

%6 step algorithm

A0 = -sqrt(3)*(I_2_0 + I_3_0 - I_5_0 - I_6_0);
B0 = 2*I_1_0 + I_2_0 - I_3_0 - 2*I_4_0 - I_5_0 + I_6_0;

for i=1:m
     for j=1:n
         phi_0(i,j)= atan2(B0(i,j),A0(i,j));
         
     end
end

 
phi_0 = phi_0 - Mean_lift(ones(m,n));

%%%%%%%%%%%%%%%%%%%%%%%%%%%Unwrapping Method 3%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Noise reduction Filter

%Unwrap object phi
delta_phi_o = phi-phi_0;

delta_phi_o(isnan(delta_phi_o))=0; %set all NaN value in deltaphi =0

delta_phi_a = delta_phi_o;

%%%noise prefilter
alpha = pi/2;
beta = 1;
gamma = 0.5;
windowsize = 9;


for i = 5:1018
    for j = 5:1275
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
                    if abs(window(p,q+1) - window(p,q)) > pi & abs(window(p,q) - window(p,q-1)) > pi
                        window(p,q) = 0;
                    elseif abs(window(p+1,q) - window(p)) > pi & abs(window(p,q) - window(p-1,q)) > pi
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
                    


unwrap_img_after1 = unwrap_phase(delta_phi_a);
unwrap_img_after2 = unwrap_phase(unwrap_img_after1);

unwrap_img_after3 = medfilt2(unwrap_img_after2,[10 10]);
unwrap_img_after4 = unwrap_phase(unwrap_img_after3);


%find maximal points which indicates coin height
C = unwrap_img_after4;
%TFc = islocalmax(C);
%[r,c] = find(TF);
%min_valuec = C(TFc);
%tfc = min_valuec > 0.5; 

phase_heightarray = C(C>0.5);
mean_height = mean(phase_heightarray,'all');
stand_dev = std(phase_heightarray,0,'all');


figure(77)
imshow(unwrap_img_after4,[]);
figure(78);
mesh(unwrap_img_after4);


%3D info
[x,y] = meshgrid(0:52/686:1279*52/686,0:51/719:1023*51/719);
figure(79);
mesh(x,y,unwrap_img_after4);
ax = gca;
ax.FontSize = 25;
xlabel('x(mm)','FontSize',25);
ylabel('y(mm)','FontSize',25);
zlabel('phi(rad)','FontSize',25);


toc
%}

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

