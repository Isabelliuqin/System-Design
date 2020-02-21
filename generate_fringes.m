
clc;
clear all;

tic
Ce = cell(1,6);  
f= 20;%number of fringes across the whole image
width = 600;  
height =800; %number of pixels
 

for j=1:6
    Ce{1,j} = zeros(width,height);%generate a cell of size 1*6 with each cell contains 500*500 size matrix
    
end

for  j = 0:5 
     for k = 1:width %row number or y
          for q=1:height %column number or x
               Ce{1,j+1}(k,q) =0.5+0.5*cos(2*pi*q*f/(height)+j*pi/3); %for each i, generate 4 patterns with different phase shift 2pif/height = k as f/heigth = 1/lambda
          end  %we generated 3 sets of 4 patterns stored in cell with different fringe spacing indicated in f(i)
     end
end

 
for j=1:6
    tmp=Ce{1,j};
         
    filename=['D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes\fringe',num2str(j),'.bmp'];
        
           %filename=['D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes',num2str(2.^(i)+j),'.bmp']; 
  
    imwrite(tmp,filename,'bmp');
end


%fringe display
second = 15;
I1 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes\fringe1.bmp');
figure(1);
imshow(I1);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');

pause(second);

I2 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes\fringe2.bmp');
figure(2);
imshow(I2);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I3 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes\fringe3.bmp');
figure(3);
imshow(I3);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I4 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes\fringe4.bmp');
figure(4);
imshow(I4);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I5 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes\fringe5.bmp');
figure(5);
imshow(I5);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I6 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\generate fringes\fringe6.bmp');
figure(6);
imshow(I6);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);
toc