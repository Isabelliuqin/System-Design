
% Algoritmos para Phase Shifting

clear all;
close all;
clc

%{
location = 'C:\Users\Isabel\Documents\MATLAB\System design\*.bmp';       %  folder in which your images exists
ds = imageDatastore(location)         %  Creates a datastore for all images in your folder

while hasdata(ds) 
    img = read(ds) ;             % read image from datastore
    figure, imshow(img);    % creates a new window for each image
end

%}

%%%read phi with object
I_1= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\fringe 5.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1);
I_2= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\fringe 6.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\fringe 7.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\fringe 8.bmp');

 
I_1=mat2gray((I_1), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2=mat2gray((I_2), [0 100000]);
I_3=mat2gray((I_3), [0 100000]);
I_4=mat2gray((I_4), [0 100000]);


%Ip=(I_1 + I_2 + I_3)/3;
%Ipp= (sqrt((3*((I_1 - I_3)^2)) + (((2*I_2) - I_1 - I_3)^2))/3);
%gamma= Ipp / Ip

%3 step algorithm
%A=(sqrt(3)*( I_1 - I_3 ));
%B=((2*I_2)- I_1 - I_3);



%4 step algorithm
A=(I_4 - I_2 );
B=(I_1 - I_3);
 
%  i=[1:0.001:];
%  j=[1:0.001:640];
%  
 for i=1:m
     for j=1:n
         phi(i,j)= atan2(B(i,j),A(i,j));
%          
%          Ip=(I_1(i,j) + I_2(i,j) + I_3(i,j))/3;
%          Ipp= (sqrt((3*((I_1(i,j) - I_3(i,j))^2)) + (((2*I_2(i,j)) - I_1(i,j) - I_3(i,j))^2))/3);
%          
%          ip(i,j)=(I_1(i,j) + I_2(i,j) + I_3(i,j))/3;
%          ipp(i,j)= (sqrt((3*((I_1(i,j) - I_3(i,j))^2)) + (((2*I_2(i,j)) - I_1(i,j) - I_3(i,j))^2))/3);
%          
%          gamma(i,j)= Ipp / Ip;
%          chupa(i,j)=phi(i,j)+gamma(i,j);
         
         %pause(0.3)
     end
 end
 
%iT=ip+ipp;

 
%figure(102);imshow(mat2gray(im2double(gamma)))
figure(4);imshow(mat2gray(im2double(phi)), 'DisplayRange', []);%im2double(I) converts the image I to double precision
% figura=imcrop(figure(4))
% saveas(gcf,'wraped.png')

%%%read phi without object
I_1_0= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\only fringe 5.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1);
I_2_0= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\only fringe 6.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3_0= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\only fringe 7.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4_0= imread('C:\Users\Isabel\Documents\MATLAB\System design\27012020\only fringe 8.bmp');

 
I_1_0=mat2gray((I_1_0), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2_0=mat2gray((I_2_0), [0 100000]);
I_3_0=mat2gray((I_3_0), [0 100000]);
I_4_0=mat2gray((I_4_0), [0 100000]);


%Ip=(I_1 + I_2 + I_3)/3;
%Ipp= (sqrt((3*((I_1 - I_3)^2)) + (((2*I_2) - I_1 - I_3)^2))/3);
%gamma= Ipp / Ip

%3 step algorithm
%A=(sqrt(3)*( I_1 - I_3 ));
%B=((2*I_2)- I_1 - I_3);



%4 step algorithm
A0=(I_4_0 - I_2_0 );
B0=(I_1_0 - I_3_0);
 
%  i=[1:0.001:];
%  j=[1:0.001:640];
%  
 for i=1:m
     for j=1:n
         phi_0(i,j)= atan2(B0(i,j),A0(i,j));
%          
%          Ip=(I_1(i,j) + I_2(i,j) + I_3(i,j))/3;
%          Ipp= (sqrt((3*((I_1(i,j) - I_3(i,j))^2)) + (((2*I_2(i,j)) - I_1(i,j) - I_3(i,j))^2))/3);
%          
%          ip(i,j)=(I_1(i,j) + I_2(i,j) + I_3(i,j))/3;
%          ipp(i,j)= (sqrt((3*((I_1(i,j) - I_3(i,j))^2)) + (((2*I_2(i,j)) - I_1(i,j) - I_3(i,j))^2))/3);
%          
%          gamma(i,j)= Ipp / Ip;
%          chupa(i,j)=phi(i,j)+gamma(i,j);
         
         %pause(0.3)
     end
 end
 
%iT=ip+ipp;

 
%figure(102);imshow(mat2gray(im2double(gamma)))
%figure(40);imshow(mat2gray(im2double(phi_0)), 'DisplayRange', []);%im2double(I) converts the image I to double precision
% figura=imcrop(figure(4))
% saveas(gcf,'wraped.png')

 

%{
%%%%%%%%%Unwrapping Method 1%%%%%%%%%%%%%%%%% 
%Unwrap the imaage using the Itoh algorithm: the first method is performed
%by first sequentially unwrapping the all rows, one at a time.




cut=imread('23012020wraped.png');

%cut =imcrop(imshow(im2double(phi)));%displays the image I in a figure window and creates an interactive Crop Image tool associated with the image

%image1_unwrapped = cut(:,:);%cut(:,:) return the same matrix as cut

%cutsize=size(cut);%size(A) returns a row vector whose elements are the lengths of the corresponding dimensions of A. For example, if A is a 3-by-4 matrix, then size(A) returns the vector [3 4].

%l=0;
%p=0;

%l=cutsize(1,1);
%p=cutsize(1,2);

%N=l;
%M=p;

image1_unwrapped = phi;
%N=m;
%M=n;
 
for i=1:n
 image1_unwrapped(:,i) = unwrap(image1_unwrapped(:,i));
 end 
for i=1:m
 image1_unwrapped(i,:) = unwrap(image1_unwrapped(i,:));
 end
 %Then sequentially unwrap all the columns one at a time
 
image1_unwrapped0 = phi_0;
%N=m;
%M=n;
 
for i=1:n
 image1_unwrapped0(:,i) = unwrap(image1_unwrapped0(:,i));
 end 
for i=1:m
 image1_unwrapped0(i,:) = unwrap(image1_unwrapped0(i,:));
 end 
 

figure(5);
imshow(image1_unwrapped,[])

figure(50);
imshow(image1_unwrapped0,[])

ave_col1 = mean(image1_unwrapped(:,1));
ave_col1_0 = mean(image1_unwrapped0(:,1));

a = ave_col1;
Ave_col1 = a(ones(m, n));

b = ave_col1_0;
Ave_col1_0 = b(ones(m, n));

image1_unwrapped = image1_unwrapped - ave_col1;
image1_unwrapped0 = image1_unwrapped0 - ave_col1_0;

delta_phi = image1_unwrapped-image1_unwrapped0;

[xidx, yidx]= find(delta_phi>25.6690);


figure(11)
imshow(delta_phi,[])
%}

%{
%%%%%%%%%%%%%%%%%%%%Unwrapping Method 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
q=im2double(image1_unwrapped);
q(400:end,:)=unwrap(q(400:end,:),[],1);
q=flipud(q);
q(400:end,:)=unwrap(q(400:end,:),[],1);
q(:,640:end)=unwrap(q(:,640:end),[],2);
q=fliplr(q);
q(:,640:end)=unwrap(q(:,640:end),[],2);
q=flipud(fliplr(q));
figure(6);
imshow(q,[]);

 
figure(7); colormap(gray(256)), imagesc(image1_unwrapped)
title('Unwrapped phase image using the Itoh algorithm: the first method')
xlabel('Pixels'), ylabel('Pixels')

%figure(8);mesh(image1_unwrapped,'FaceColor','interp', 'EdgeColor','none', 'FaceLighting','phong')
view(-30,30), camlight left, axis tight
title('Unwrapped phase image using the Itoh algorithm: the first method')
xlabel('Pixels'), ylabel('Pixels'), zlabel('Phase in radians')

%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%Unwrapping Method 3%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
%%%Test phasewrapping
fprintf('***************************************\n');
fprintf('2D Phase Unwrapping Demo\n');
fprintf('Please select the demo:\n');
fprintf('(1) No noise  , no ignored region\n');
fprintf(' 2. With noise, no ignored region\n');
fprintf(' 3. No noise  , with ignored region\n');
fprintf(' 4. With noise, with ignored region\n');
while (1)
    user_input = input('Your selection (1-4): ', 's');
    user_input = strip(user_input);

    % if the user does not supply anything, select the default
    if strcmp(user_input, '')
        fprintf('Demo 1 is selected\n');
        user_input = '1';
    end

    if length(user_input) == 1 && sum(user_input == '1234') == 1
        break;
    else
        fprintf('Invalid input\n');
    end
end


[X, Y] = meshgrid(linspace(-1, 1, 512) * 5);
img = -(X.*X + Y.*Y);
fprintf('Image size: %dx%d pixels\n', size(img,1), size(img,2));

% add noise
if any(user_input == '24')
    img = img + randn(size(X)) * 0.5;
end

% add an ignored region
if any(user_input == '34')
    img(end/4:3*end/4,end/4:3*end/4) = nan;
end

% wrap the image
wimg = wrapTo2Pi(img);

tic;
unwrap_img = unwrap_phase(wimg);
toc;

subplot(221)
pcolor(img)
shading flat;
set(gca, 'ydir', 'reverse');
title('Original phase');

subplot(222)
pcolor(wimg)
shading flat;
set(gca, 'ydir', 'reverse');
title('Wrapped phase');

subplot(223)
pcolor(unwrap_img)
shading flat;
set(gca, 'ydir', 'reverse');
title('Unwrapped phase');

subplot(224)
pcolor(wrapTo2Pi(unwrap_img))
shading flat;
set(gca, 'ydir', 'reverse');
title('Rewrap of unwrapped phase');
%}


%%%method
tic;
unwrap_img = unwrap_phase(phi);
toc;


unwrap_img_0 = unwrap_phase(phi_0);


%figure(9)
imshow(unwrap_img,[])
%pcolor(unwrap_img)
shading flat;
set(gca, 'ydir', 'reverse');
title('Wrapped phase');


%figure(90)
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

unwrap_img = unwrap_img - ave_col1;
unwrap_img_0 = unwrap_img_0 - ave_col1_0;


delta_phi = unwrap_img - unwrap_img_0;

unwrap_img = unwrap_phase(delta_phi);
%unwrap_img = unwrap_phase(unwrap_img);

%{
for i=1:n
 image1_unwrapped(:,i) = unwrap(image1_unwrapped(:,i));
 end 
for i=1:m
 image1_unwrapped(i,:) = unwrap(image1_unwrapped(i,:));
end
%} 
figure(10)
imshow(unwrap_img,[])

figure(20),mesh(unwrap_img)


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





