

filepath = 'image6.png';


p.I = imread(filepath);
    p.dx = (p.X-1)/p.cx;            
    p.dy = (p.Y-1)/p.cy;
    p.X = size(p.I, 2);
    p.Y = size(p.I, 1);    
    p.cx = floor(size(p.I,2)/30);
    p.cy = floor(size(p.I,1)/30);
    p.NV = (p.cx+1)*(p.cy+1);
    p.vertexX = 1:p.dx:p.X;     
    p.vertexY = 1:p.dy:p.Y;
    [p.gridX, p.gridY] = meshgrid(p.vertexX, p.vertexY);
    p.Vx = reshape(p.gridX, p.NV, 1);
    p.Vy = reshape(p.gridY, p.NV, 1);
    p.k = 1:p.NV;


%imshow(p.I)                    % Commented to ensure that the code is used
                                   % for displaying the initial mesh
%hold on;
%%gx = reshape(p.Vx, p.cy+1, p.cx+1);
%gy = reshape(p.Vy, p.cy+1, p.cx+1);
%plot(gx, gy, 'y-', gx', gy', 'y-');

V = csvread('Vertices6.csv');
V = V(:,2);
V = V(2:size(V));
Vx = V(2*p.k-1);
Vy = V(2*p.k);
I = warpMesh(p,Vx,Vy);
imshow(I)
Vx1 = Vx;
Vy1 = Vy;
%imwrite(I, '/Current Results/image11.jpg');
    hold on;
gx = reshape(Vx1, p.cy+1, p.cx+1);
gy = reshape(Vy1, p.cy+1, p.cx+1);
plot(gx, gy, 'y-', gx', gy', 'y-');
hold off;



