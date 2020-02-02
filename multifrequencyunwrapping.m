clear all;
close all;
clc

%fringe period lambda h

%%%read phi with object
I_1high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 9 2.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1high);
I_2high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 10 2.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 11 2.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 12 2.bmp');

 
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
              %pause(0.3)
     end
end






%%%read phi without object
I_1_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 9 2.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1_0high);
I_2_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 10 2.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 11 2.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4_0high= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 12 2.bmp');

 
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
I_1low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 9 2.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1low);
I_2low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 10 2.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 11 2.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\fringe 12 2.bmp');

 
I_1low=mat2gray((I_1low), [0 100000]); %mat2gray converts the matrix to an intensity image I that contains values in the range 0 (black) to 1 (white). amin and amax are the values in A that correspond to 0 and 1 in I. Values less than amin become 0, and values greater than amax become 1.
I_2low=mat2gray((I_2low), [0 100000]);
I_3low=mat2gray((I_3low), [0 100000]);
I_4low=mat2gray((I_4low), [0 100000]);


%4 step algorithm
Alow=(I_4low - I_2low );
Blow=(I_1low - I_3low);

for i=1:m
     for j=1:n
         phi_low(i,j)= atan2(B(i,j),A(i,j));
              %pause(0.3)
     end
end






%%%read phi without object
I_1_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 9 2.bmp');
%I_1= imread('Sample 1.bmp');
 %figure(1); imshow(I_1)
 %title ('Imagen de intensidad 1')
[m n] = size(I_1low);
I_2_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 10 2.bmp');
%I_2= imread('Sample 2.bmp');
 %figure(2); imshow(I_2)
 %title('Imagen de intensidad 2')
 
I_3_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 11 2.bmp');
%I_3= imread('Sample 3.bmp');
 %figure(3); imshow(I_3)
 %title('Imagen de intensidad 3')
I_4_0low= imread('D:\IC\Master degree\Laboratory\System Design\experiment\31012020\only fringe 12 2.bmp');

 
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

delta_phi_eq = delta_phi_a_high - delta_phi_a_low;
lambda_eq = (lambda_low * lambda_high)/(lambda_low - lambda_high);
k_h = round(((lambda_eq/lambda_h)*delta_phi_eq - delta_phi_a_high)/(2*pi));

delta_phi_unwrapped = delta_phi_a_high + 2*pi*delta_phi_eq;