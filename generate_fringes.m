
clc;
clear all;
Ce = cell(2,4);  
f= [50 55];%number of fringes across the whole image
width = 500;  
heigth =500; %number of pixels
 
%?
for i=1:2
    for j=1:4
        Ce{i,j} = zeros(width,heigth);%generate a cell of size 3*4 with each cell contains 500*500 size matrix
    end
end
for i = 1:2 % ?????????
    for  j = 0:3 % ??????
        for k = 1:width %row number or y
            for q=1:heigth %column number or x
               Ce{i,j+1}(k,q) =0.5+0.5*cos(2*pi*q*f(i)/(heigth)+j*pi/2); %for each i, generate 4 patterns with different phase shift 2pif/height = k as f/heigth = 1/lambda
            end  %we generated 3 sets of 4 patterns stored in cell with different fringe spacing indicated in f(i)
        end
    end
end

for i = 1:2  %????12??????
     for j=1:4
         tmp=Ce{i,j};
         if i==1
           filename=['D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe',num2str(j),'.bmp'];
         elseif i>1
           filename=['D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe',num2str(2.^(i)+j),'.bmp']; 
         end
         imwrite(tmp,filename,'bmp');
     end
end

%fringe display
second = 15
I1 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe1.bmp');
figure(1);
imshow(I1);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');

pause(second);

I2 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe2.bmp');
figure(2);
imshow(I2);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I3 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe3.bmp');
figure(3);
imshow(I3);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I4 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe4.bmp');
figure(4);
imshow(I4);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I5 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe5.bmp');
figure(5);
imshow(I5);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I6 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe6.bmp');
figure(6);
imshow(I6);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I7 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe7.bmp');
figure(7);
imshow(I7);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);

I8 = imread('D:\IC\Master degree\Laboratory\System Design\experiment\10022020\fringe7.bmp');
figure(8);
imshow(I8);

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
pause(second);