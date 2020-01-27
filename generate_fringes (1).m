
%%%??3??12????
%%%?????73 64 56

clc;
clear all;
Ce = cell(3,4);  
f= [20 30 40];
width = 500;  
heigth =500;
 
%?
for i=1:3
    for j=1:4
        Ce{i,j} = zeros(width,heigth);
    end
end
for i = 1:3 % ?????????
    for  j = 0:3 % ??????
        for k = 1:width 
            for q=1:heigth
              Ce{i,j+1}(k,q) =0.5+0.5*cos(2*pi*q*f(i)/(heigth)+j*pi/2);
            end
        end
    end
end
for i = 1:3  %????12??????
     for j=1:4
         tmp=Ce{i,j};
         if i==1
           filename=['\\icnas3.cc.ic.ac.uk\wz1416\Desktop\Profiling system\generate fringes\fringes',num2str(j),'.bmp'];
         elseif i>1
           filename=['\\icnas3.cc.ic.ac.uk\wz1416\Desktop\Profiling system\generate fringes\fringes',num2str(2.^(i)+j),'.bmp']; 
         end
         imwrite(tmp,filename,'bmp');
     end
end